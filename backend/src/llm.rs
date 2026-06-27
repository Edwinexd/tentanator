//! LLM provider clients: OpenAI embeddings (for maximin sampling) and Cerebras
//! chat completions (for in-context-learning grade suggestions).

use std::time::Duration;

use serde_json::{json, Value};

use crate::config::Config;
use crate::domain::{is_meaningful, AIGradeSuggestion, GradedItem, QuestionGrades};
use crate::error::{AppError, AppResult};
use crate::grade::evaluate_grade;

pub const MIN_ICL_EXAMPLES: usize = 5;
pub const MAX_ICL_EXAMPLES: usize = 25;

/// Longest student answer text fed into the prompt (examples and the target).
const MAX_ANSWER_CHARS: usize = 1000;

/// Retry budget for transient upstream failures (429 / 5xx / connect errors).
const MAX_RETRIES: u32 = 4;

/// Cap on concurrent embedding requests so a large reindex can't exhaust the
/// connection pool or trip rate limits.
const EMBED_CONCURRENCY: usize = 8;

const BASE_SYSTEM_PROMPT: &str = "You are an experienced teacher assistant helping grade student exam responses.
Your task is to evaluate the student's answer to the following question and provide a grade.
Be consistent, fair, and objective in your grading. All responses will be reviewed by a human teacher.

Exam Question: {exam_question}

Grading Criteria:
- Correctness of the answer
- Completeness of the response

Provide only the grade value as your response: a number, or a signed sum of subpart \
scores (e.g. 5, 7.5, 2+1.5+2.5). Output nothing but the grade.";

/// Backoff delay before retry `attempt` (0-based): 0.5s, 1s, 2s, 4s.
fn backoff(attempt: u32) -> Duration {
    Duration::from_millis(500u64 << attempt.min(5))
}

/// Parse a `Retry-After` header given in whole seconds.
fn retry_after(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    headers
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()
        .map(Duration::from_secs)
}

/// POST a JSON body with bounded retry on 429/5xx and transient connect errors.
/// `label` only tags log/error messages.
async fn post_json_with_retry(
    client: &reqwest::Client,
    url: &str,
    api_key: &str,
    body: &Value,
    label: &str,
) -> AppResult<Value> {
    let mut attempt = 0u32;
    loop {
        match client.post(url).bearer_auth(api_key).json(body).send().await {
            Ok(resp) => {
                let status = resp.status();
                if status.is_success() {
                    return resp.json::<Value>().await.map_err(|e| AppError::Upstream(e.to_string()));
                }
                let retryable =
                    status == reqwest::StatusCode::TOO_MANY_REQUESTS || status.is_server_error();
                let wait = retry_after(resp.headers());
                let detail = resp.text().await.unwrap_or_default();
                if retryable && attempt < MAX_RETRIES {
                    let delay = wait.unwrap_or_else(|| backoff(attempt));
                    tracing::warn!("{label} {status} (attempt {attempt}), retrying in {delay:?}");
                    tokio::time::sleep(delay).await;
                    attempt += 1;
                    continue;
                }
                return Err(AppError::Upstream(format!("{label} {status}: {detail}")));
            }
            Err(e) => {
                let transient = e.is_timeout() || e.is_connect() || e.is_request();
                if transient && attempt < MAX_RETRIES {
                    let delay = backoff(attempt);
                    tracing::warn!("{label} request error (attempt {attempt}), retrying in {delay:?}: {e}");
                    tokio::time::sleep(delay).await;
                    attempt += 1;
                    continue;
                }
                return Err(AppError::Upstream(e.to_string()));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Embeddings
// ---------------------------------------------------------------------------

pub async fn embed(config: &Config, client: &reqwest::Client, text: &str) -> AppResult<Vec<f32>> {
    let cleaned = text.replace('\n', " ");
    let url = format!("{}/embeddings", config.openai_base_url.trim_end_matches('/'));
    let body = json!({ "input": cleaned, "model": config.embedding_model });
    let v = post_json_with_retry(client, &url, &config.openai_api_key, &body, "embeddings").await?;
    let arr = v
        .pointer("/data/0/embedding")
        .and_then(|e| e.as_array())
        .ok_or_else(|| AppError::Upstream("embeddings response missing data".into()))?;
    Ok(arr.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect())
}

/// Embed many texts with bounded concurrency. Returns `None` for any that
/// failed; results stay aligned with the input order.
pub async fn embed_many(
    config: &Config,
    client: &reqwest::Client,
    texts: &[String],
) -> Vec<Option<Vec<f32>>> {
    use futures::stream::StreamExt;
    let futs = texts
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, t)| async move { (i, embed(config, client, &t).await.ok()) });
    let mut out: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
    let results = futures::stream::iter(futs)
        .buffer_unordered(EMBED_CONCURRENCY)
        .collect::<Vec<_>>()
        .await;
    for (i, r) in results {
        out[i] = r;
    }
    out
}

// ---------------------------------------------------------------------------
// In-context-learning grade suggestion
// ---------------------------------------------------------------------------

fn build_icl_messages(question: &QuestionGrades, response_text: &str) -> Vec<Value> {
    let system_content = if !question.exam_question.is_empty() {
        BASE_SYSTEM_PROMPT.replace("{exam_question}", &question.exam_question)
    } else {
        format!("You are grading responses for: {}", question.question_name)
    };
    let mut messages = vec![json!({ "role": "system", "content": system_content })];

    let mut examples: Vec<&GradedItem> = question
        .icl_candidates()
        .into_iter()
        .filter(|i| is_meaningful(&i.input_text))
        .collect();

    // Evenly sample across the grade range when over the cap. Sort by grade with
    // a stable row_id tiebreak so the selection is deterministic across loads,
    // and span the full range inclusively so the top grade isn't dropped.
    if examples.len() > MAX_ICL_EXAMPLES {
        examples.sort_by(|a, b| {
            let ga = evaluate_grade(&a.grade).unwrap_or(0.0);
            let gb = evaluate_grade(&b.grade).unwrap_or(0.0);
            ga.partial_cmp(&gb)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.row_id.cmp(&b.row_id))
        });
        let last = examples.len() - 1;
        examples = (0..MAX_ICL_EXAMPLES)
            .map(|i| examples[i * last / (MAX_ICL_EXAMPLES - 1)])
            .collect();
    }

    for item in examples {
        messages.push(json!({ "role": "user", "content": truncate_answer(&item.input_text) }));
        messages.push(json!({ "role": "assistant", "content": item.grade }));
    }
    messages.push(json!({ "role": "user", "content": truncate_answer(response_text) }));
    messages
}

/// Cap an answer at [`MAX_ANSWER_CHARS`] so one long response can't blow the
/// model's context window.
fn truncate_answer(text: &str) -> String {
    if text.chars().count() > MAX_ANSWER_CHARS {
        text.chars().take(MAX_ANSWER_CHARS).collect::<String>() + "..."
    } else {
        text.to_string()
    }
}

/// True if (own + pooled) meaningful examples reach the ICL minimum.
pub fn has_enough_icl(question: &QuestionGrades) -> bool {
    question
        .icl_candidates()
        .iter()
        .filter(|i| is_meaningful(&i.input_text))
        .count()
        >= MIN_ICL_EXAMPLES
}

pub async fn suggest_grade(
    config: &Config,
    client: &reqwest::Client,
    question: &QuestionGrades,
    response_text: &str,
) -> AppResult<AIGradeSuggestion> {
    let messages = build_icl_messages(question, response_text);
    let url = format!(
        "{}/chat/completions",
        config.cerebras_base_url.trim_end_matches('/')
    );
    let body = json!({
        "model": config.cerebras_model,
        "messages": messages,
        "max_tokens": 4096,
        "temperature": 0.0,
        "reasoning_effort": config.cerebras_reasoning_effort,
    });
    let v = post_json_with_retry(client, &url, &config.cerebras_api_key, &body, "chat").await?;
    let message = v.pointer("/choices/0/message").cloned().unwrap_or(Value::Null);

    let (reasoning, answer) = extract_reasoning(&message);
    // Fail safe: an unparseable model reply yields an empty suggestion (no
    // prefill) rather than free text masquerading as a grade.
    let grade = extract_grade_from_answer(&answer).unwrap_or_default();
    let reasoning_summary = if reasoning.trim().is_empty() {
        None
    } else {
        summarize_reasoning(config, client, &reasoning).await.ok().flatten()
    };
    Ok(AIGradeSuggestion { grade, reasoning_summary })
}

async fn summarize_reasoning(
    config: &Config,
    client: &reqwest::Client,
    reasoning: &str,
) -> AppResult<Option<String>> {
    let url = format!(
        "{}/chat/completions",
        config.cerebras_base_url.trim_end_matches('/')
    );
    let system = "You condense grading rationales into a compact paragraph (3-5 sentences, \
        60-120 words). Cover: (1) which criteria or key points the answer addressed well, \
        (2) what was missing, wrong, or weak, and (3) any comparisons made to similar graded \
        examples. Never output a number, grade, score, or arithmetic expression. Do not state \
        the score value.";
    let user = format!(
        "Rationale:\n{}\n\nSummary (no numbers, 3-5 sentences, 60-120 words):",
        reasoning.trim()
    );
    let resp = client
        .post(&url)
        .bearer_auth(&config.cerebras_api_key)
        .json(&json!({
            "model": config.cerebras_summary_model,
            "messages": [
                { "role": "system", "content": system },
                { "role": "user", "content": user },
            ],
            "max_tokens": 800,
            "temperature": 0.2,
            "reasoning_effort": "low",
        }))
        .send()
        .await
        .map_err(|e| AppError::Upstream(e.to_string()))?;
    if !resp.status().is_success() {
        return Ok(None);
    }
    let v: Value = resp.json().await.map_err(|e| AppError::Upstream(e.to_string()))?;
    let summary = v
        .pointer("/choices/0/message/content")
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if summary.is_empty() || evaluate_grade(&summary).is_some() {
        return Ok(None);
    }
    Ok(Some(summary))
}

/// Returns (reasoning, answer) from a chat message value.
fn extract_reasoning(message: &Value) -> (String, String) {
    let content = message
        .get("content")
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let reasoning = message
        .get("reasoning")
        .and_then(|c| c.as_str())
        .or_else(|| message.get("reasoning_content").and_then(|c| c.as_str()))
        .unwrap_or("")
        .to_string();
    if !reasoning.is_empty() {
        return (reasoning.trim().to_string(), content);
    }
    // Fallback: inline <think>...</think> blocks.
    if let Some(think) = extract_think_blocks(&content) {
        let answer = strip_think_blocks(&content);
        return (think, answer);
    }
    (String::new(), content)
}

fn extract_think_blocks(text: &str) -> Option<String> {
    let mut blocks = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find("<think>") {
        let after = &rest[start + "<think>".len()..];
        if let Some(end) = after.find("</think>") {
            blocks.push(after[..end].trim().to_string());
            rest = &after[end + "</think>".len()..];
        } else {
            break;
        }
    }
    if blocks.is_empty() {
        None
    } else {
        Some(blocks.join("\n"))
    }
}

fn strip_think_blocks(text: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(start) = rest.find("<think>") {
        out.push_str(&rest[..start]);
        let after = &rest[start + "<think>".len()..];
        if let Some(end) = after.find("</think>") {
            rest = &after[end + "</think>".len()..];
        } else {
            rest = "";
            break;
        }
    }
    out.push_str(rest);
    out.trim().to_string()
}

/// Extract a parseable grade from the model's answer, or `None` if nothing in
/// the reply parses as a grade (so the caller can fail safe rather than emit
/// free text as a grade).
fn extract_grade_from_answer(answer: &str) -> Option<String> {
    let text = answer.trim();
    if evaluate_grade(text).is_some() {
        return Some(text.to_string());
    }
    let last_line = text.lines().last().unwrap_or("").trim();
    if !last_line.is_empty() && evaluate_grade(last_line).is_some() {
        return Some(last_line.to_string());
    }
    let hay = if last_line.is_empty() { text } else { last_line };
    trailing_grade(hay)
}

/// Longest trailing substring that parses as a grade, whitespace removed.
fn trailing_grade(hay: &str) -> Option<String> {
    let chars: Vec<char> = hay.chars().collect();
    for start in 0..chars.len() {
        let cand: String = chars[start..].iter().collect();
        let trimmed = cand.trim();
        if trimmed.is_empty() {
            continue;
        }
        let first = trimmed.chars().next().unwrap();
        if !(first.is_ascii_digit() || first == '+' || first == '-' || first == '.') {
            continue;
        }
        if evaluate_grade(trimmed).is_some() {
            return Some(trimmed.chars().filter(|c| !c.is_whitespace()).collect());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grade_from_plain() {
        assert_eq!(extract_grade_from_answer("7.5").as_deref(), Some("7.5"));
    }

    #[test]
    fn grade_from_trailing() {
        assert_eq!(extract_grade_from_answer("The grade is 2+3").as_deref(), Some("2+3"));
        assert_eq!(extract_grade_from_answer("Final score:\n8").as_deref(), Some("8"));
    }

    #[test]
    fn unparseable_is_none() {
        assert_eq!(extract_grade_from_answer("I cannot grade this answer."), None);
        assert_eq!(extract_grade_from_answer(""), None);
    }

    #[test]
    fn think_block_extraction() {
        let msg = json!({ "content": "<think>reasoning here</think>5" });
        let (r, a) = extract_reasoning(&msg);
        assert_eq!(r, "reasoning here");
        assert_eq!(a, "5");
    }
}
