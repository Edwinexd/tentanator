//! LLM provider clients: OpenAI embeddings (for maximin sampling) and Cerebras
//! chat completions (for in-context-learning grade suggestions).

use serde_json::{json, Value};

use crate::config::Config;
use crate::domain::{is_meaningful, AIGradeSuggestion, GradedItem, QuestionGrades};
use crate::error::{AppError, AppResult};
use crate::grade::evaluate_grade;

pub const MIN_ICL_EXAMPLES: usize = 5;
pub const MAX_ICL_EXAMPLES: usize = 25;

const BASE_SYSTEM_PROMPT: &str = "You are an experienced teacher assistant helping grade student exam responses.
Your task is to evaluate the student's answer to the following question and provide a grade.
Be consistent, fair, and objective in your grading. All respones will be reviewed by a human teacher.

Exam Question: {exam_question}

Grading Criteria:
- Correctness of the answer
- Completeness of the response

Provide only the grade value as your response.";

// ---------------------------------------------------------------------------
// Embeddings
// ---------------------------------------------------------------------------

pub async fn embed(config: &Config, client: &reqwest::Client, text: &str) -> AppResult<Vec<f32>> {
    let cleaned = text.replace('\n', " ");
    let url = format!("{}/embeddings", config.openai_base_url.trim_end_matches('/'));
    let resp = client
        .post(&url)
        .bearer_auth(&config.openai_api_key)
        .json(&json!({ "input": cleaned, "model": config.embedding_model }))
        .send()
        .await
        .map_err(|e| AppError::Upstream(e.to_string()))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(AppError::Upstream(format!("embeddings {status}: {body}")));
    }
    let v: Value = resp.json().await.map_err(|e| AppError::Upstream(e.to_string()))?;
    let arr = v
        .pointer("/data/0/embedding")
        .and_then(|e| e.as_array())
        .ok_or_else(|| AppError::Upstream("embeddings response missing data".into()))?;
    Ok(arr.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect())
}

/// Embed many texts concurrently. Returns `None` for any that failed.
pub async fn embed_many(
    config: &Config,
    client: &reqwest::Client,
    texts: &[String],
) -> Vec<Option<Vec<f32>>> {
    let futs = texts.iter().map(|t| embed(config, client, t));
    futures::future::join_all(futs)
        .await
        .into_iter()
        .map(|r| r.ok())
        .collect()
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

    // Evenly sample across the grade range when over the cap.
    if examples.len() > MAX_ICL_EXAMPLES {
        examples.sort_by(|a, b| {
            let ga = evaluate_grade(&a.grade).unwrap_or(0.0);
            let gb = evaluate_grade(&b.grade).unwrap_or(0.0);
            ga.partial_cmp(&gb).unwrap_or(std::cmp::Ordering::Equal)
        });
        let step = examples.len() as f64 / MAX_ICL_EXAMPLES as f64;
        examples = (0..MAX_ICL_EXAMPLES)
            .map(|i| examples[(i as f64 * step) as usize])
            .collect();
    }

    for item in examples {
        let mut text = item.input_text.clone();
        if text.chars().count() > 1000 {
            text = text.chars().take(1000).collect::<String>() + "...";
        }
        messages.push(json!({ "role": "user", "content": text }));
        messages.push(json!({ "role": "assistant", "content": item.grade }));
    }
    messages.push(json!({ "role": "user", "content": response_text }));
    messages
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
    let resp = client
        .post(&url)
        .bearer_auth(&config.cerebras_api_key)
        .json(&json!({
            "model": config.cerebras_model,
            "messages": messages,
            "max_tokens": 4096,
            "temperature": 0.0,
            "reasoning_effort": config.cerebras_reasoning_effort,
        }))
        .send()
        .await
        .map_err(|e| AppError::Upstream(e.to_string()))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(AppError::Upstream(format!("chat {status}: {body}")));
    }
    let v: Value = resp.json().await.map_err(|e| AppError::Upstream(e.to_string()))?;
    let message = v.pointer("/choices/0/message").cloned().unwrap_or(Value::Null);

    let (reasoning, answer) = extract_reasoning(&message);
    let grade = extract_grade_from_answer(&answer);
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

fn extract_grade_from_answer(answer: &str) -> String {
    let text = answer.trim();
    if evaluate_grade(text).is_some() {
        return text.to_string();
    }
    let last_line = text.lines().last().unwrap_or("").trim();
    if !last_line.is_empty() && evaluate_grade(last_line).is_some() {
        return last_line.to_string();
    }
    let hay = if last_line.is_empty() { text } else { last_line };
    if let Some(g) = trailing_grade(hay) {
        return g;
    }
    if last_line.is_empty() {
        text.to_string()
    } else {
        last_line.to_string()
    }
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
        assert_eq!(extract_grade_from_answer("7.5"), "7.5");
    }

    #[test]
    fn grade_from_trailing() {
        assert_eq!(extract_grade_from_answer("The grade is 2+3"), "2+3");
        assert_eq!(extract_grade_from_answer("Final score:\n8"), "8");
    }

    #[test]
    fn think_block_extraction() {
        let msg = json!({ "content": "<think>reasoning here</think>5" });
        let (r, a) = extract_reasoning(&msg);
        assert_eq!(r, "reasoning here");
        assert_eq!(a, "5");
    }
}
