//! Global question bank: an app-wide (not exam/course-scoped) library of
//! reference questions used to (a) semantically search past questions and
//! (b) auto-match the question a column is grading, pre-filling its text /
//! sample answer / `global_question_id` so cross-exam ICL kicks in.
//!
//! The bank lives in the database (`global_bank_questions`), managed in-app:
//! questions are imported from a CSV through `POST /api/global-bank/import`,
//! never fetched from a remote URL. `q_se` / `q_en` are embedded with the same
//! OpenAI model used for sampling and cached in `global_bank_vectors`.
//!
//! Ported from the legacy `process_global.py` / `search_question.py` and the
//! `find_best_matching_questions` flow in `tentanator.py`. Deviation: language
//! detection is a dependency-free heuristic (Swedish-specific letters) rather
//! than a gpt-4o-mini call; callers may override it explicitly.

use std::collections::HashMap;
use std::path::Path;

use serde::Serialize;
use ts_rs::TS;

use crate::domain::BankQuestion;
use crate::error::AppResult;
use crate::store;

/// A ranked bank match returned to the clients.
#[derive(Clone, Debug, Serialize, TS)]
#[ts(export, export_to = "../../web/src/lib/generated/")]
pub struct GlobalBankMatch {
    pub bank: String,
    pub qid: String,
    pub score: f64,
    pub q_se: String,
    pub q_en: String,
    pub ans_se: String,
    pub ans_en: String,
    pub chapter: String,
    pub subject: String,
    pub qtype: String,
}

impl GlobalBankMatch {
    fn from(q: &BankQuestion, score: f64) -> Self {
        Self {
            bank: q.bank.clone(),
            qid: q.qid.clone(),
            score,
            q_se: q.q_se.clone(),
            q_en: q.q_en.clone(),
            ans_se: q.ans_se.clone(),
            ans_en: q.ans_en.clone(),
            chapter: q.chapter.clone(),
            subject: q.subject.clone(),
            qtype: q.qtype.clone(),
        }
    }
}

/// Case-insensitive field lookup from a parsed sheet row.
fn field(row: &HashMap<String, String>, name: &str) -> String {
    row.iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(name))
        .map(|(_, v)| v.trim().to_string())
        .unwrap_or_default()
}

/// Parse a bank CSV/xlsx into questions (for import into the DB). `bank` labels
/// the source; rows without an `id` are skipped. Columns: `id`, `q_se`, `q_en`,
/// `ans_se`, `ans_en`, `chapter`, `subject`, `type` (case-insensitive).
pub fn parse_bank_csv(path: &Path, bank: &str) -> AppResult<Vec<BankQuestion>> {
    let rows = store::read_exam_data(path)?;
    let mut out = Vec::new();
    for row in &rows {
        let qid = field(row, "id");
        if qid.is_empty() {
            continue;
        }
        out.push(BankQuestion {
            bank: bank.to_string(),
            qid,
            q_se: field(row, "q_se"),
            q_en: field(row, "q_en"),
            ans_se: field(row, "ans_se"),
            ans_en: field(row, "ans_en"),
            chapter: field(row, "chapter"),
            subject: field(row, "subject"),
            qtype: field(row, "type"),
        });
    }
    Ok(out)
}

/// Heuristic language detection: Swedish-specific letters => `se`, else `en`.
pub fn detect_language(text: &str) -> &'static str {
    if text
        .chars()
        .any(|c| matches!(c, 'å' | 'ä' | 'ö' | 'Å' | 'Ä' | 'Ö'))
    {
        "se"
    } else {
        "en"
    }
}

/// Normalise a requested language to `se`/`en` (default `en`).
pub fn norm_lang(lang: Option<&str>) -> &'static str {
    match lang.map(str::trim) {
        Some("se") => "se",
        _ => "en",
    }
}

pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b).map(|(x, y)| (*x as f64) * (*y as f64)).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// Mean of equal-length vectors (the centroid used to match student answers).
pub fn centroid(vectors: &[Vec<f32>]) -> Vec<f32> {
    let mut iter = vectors.iter().filter(|v| !v.is_empty());
    let Some(first) = iter.next() else {
        return Vec::new();
    };
    let mut sum = first.clone();
    let mut n = 1usize;
    for v in iter {
        if v.len() != sum.len() {
            continue;
        }
        for (s, x) in sum.iter_mut().zip(v) {
            *s += *x;
        }
        n += 1;
    }
    for s in &mut sum {
        *s /= n as f32;
    }
    sum
}

/// Rank bank questions by cosine similarity of `query` against the cached
/// `lang` vectors; return the top `k`.
pub fn rank(
    query: &[f32],
    banks: &[BankQuestion],
    vectors: &HashMap<(String, String), Vec<f32>>,
    k: usize,
) -> Vec<GlobalBankMatch> {
    let mut scored: Vec<(f64, &BankQuestion)> = banks
        .iter()
        .filter_map(|q| {
            let v = vectors.get(&(q.bank.clone(), q.qid.clone()))?;
            let score = cosine(query, v);
            Some((score, q))
        })
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored
        .into_iter()
        .take(k)
        .map(|(score, q)| GlobalBankMatch::from(q, score))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn language_heuristic() {
        assert_eq!(detect_language("what is a derivative"), "en");
        assert_eq!(detect_language("vad är en derivata"), "se");
    }

    #[test]
    fn cosine_and_rank() {
        let banks = vec![
            BankQuestion {
                bank: "b".into(),
                qid: "1".into(),
                q_en: "a".into(),
                ..Default::default()
            },
            BankQuestion {
                bank: "b".into(),
                qid: "2".into(),
                q_en: "b".into(),
                ..Default::default()
            },
        ];
        let mut vectors = HashMap::new();
        vectors.insert(("b".into(), "1".into()), vec![1.0, 0.0]);
        vectors.insert(("b".into(), "2".into()), vec![0.0, 1.0]);
        let out = rank(&[1.0, 0.1], &banks, &vectors, 2);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].qid, "1"); // closest to [1, 0.1]
        assert!(out[0].score > out[1].score);
    }
}
