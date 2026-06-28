//! Heuristic that pairs `Response N` / `Points N` columns into questions and
//! guesses the id column. The backend owns it so both clients share one
//! detector instead of each reimplementing the regexes. Ported from the former
//! web client `detectQuestionPairs`.

use std::collections::HashMap;

use serde::Serialize;

#[derive(Debug, Default, Serialize)]
pub struct DetectedColumns {
    pub id_columns: Vec<String>,
    pub input_columns: Vec<String>,
    pub output_columns: Vec<String>,
}

/// Match `^response\s*(\d+)$` (case-insensitive) against an already-trimmed,
/// lowercased column name; return the captured number.
fn response_index(lo: &str) -> Option<i64> {
    let rest = lo.strip_prefix("response")?.trim_start();
    digits(rest)
}

/// Match `^points?\s*(\d+)$` (case-insensitive).
fn points_index(lo: &str) -> Option<i64> {
    let rest = lo.strip_prefix("point")?;
    let rest = rest.strip_prefix('s').unwrap_or(rest).trim_start();
    digits(rest)
}

fn digits(s: &str) -> Option<i64> {
    if !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()) {
        s.parse().ok()
    } else {
        None
    }
}

/// `/daisy\s*id/i` as a substring test.
fn has_daisy_id(c: &str) -> bool {
    let lo = c.to_lowercase();
    let mut hay = lo.as_str();
    while let Some(idx) = hay.find("daisy") {
        if hay[idx + "daisy".len()..].trim_start().starts_with("id") {
            return true;
        }
        hay = &hay[idx + "daisy".len()..];
    }
    false
}

pub fn detect_question_pairs(columns: &[String]) -> DetectedColumns {
    let mut inputs: HashMap<i64, String> = HashMap::new();
    let mut outputs: HashMap<i64, String> = HashMap::new();
    for c in columns {
        let lo = c.trim().to_lowercase();
        if let Some(n) = response_index(&lo) {
            inputs.insert(n, c.clone());
        } else if let Some(n) = points_index(&lo) {
            outputs.insert(n, c.clone());
        }
    }
    let mut ns: Vec<i64> = inputs
        .keys()
        .copied()
        .filter(|n| outputs.contains_key(n))
        .collect();
    ns.sort_unstable();

    let id = columns
        .iter()
        .find(|c| has_daisy_id(c))
        .or_else(|| columns.iter().find(|c| c.eq_ignore_ascii_case("id")))
        .or_else(|| columns.first());

    DetectedColumns {
        id_columns: id.map(|c| vec![c.clone()]).unwrap_or_default(),
        input_columns: ns.iter().map(|n| inputs[n].clone()).collect(),
        output_columns: ns.iter().map(|n| outputs[n].clone()).collect(),
    }
}

/// Sanitize an arbitrary column name into a valid scheme identifier: non
/// alphanumeric/underscore runs become `_`, and a leading digit/empty string is
/// prefixed with `_`. Lowercased so the same column always maps to one var.
pub fn sanitize_ident(s: &str) -> String {
    let mut out: String = s
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' })
        .collect::<String>()
        .to_lowercase();
    if out.is_empty() || out.starts_with(|c: char| c.is_ascii_digit()) {
        out.insert(0, '_');
    }
    out
}

/// Suggested scheme variable for a question column. `Points 3` -> `p3`,
/// `Response 3` -> `r3`; any other column is sanitized to a valid identifier.
/// The backend owns this so both clients show the same default without each
/// reimplementing the `Points N` regex.
pub fn default_var(col: &str) -> String {
    let lo = col.trim().to_lowercase();
    if let Some(n) = points_index(&lo) {
        format!("p{n}")
    } else if let Some(n) = response_index(&lo) {
        format!("r{n}")
    } else {
        sanitize_ident(col)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn default_var_derives_from_points_and_response() {
        assert_eq!(default_var("Points 1"), "p1");
        assert_eq!(default_var("Point 12"), "p12");
        assert_eq!(default_var("points3"), "p3");
        assert_eq!(default_var("Response 2"), "r2");
        assert_eq!(default_var("Daisy ID"), "daisy_id");
        assert_eq!(default_var("3a Score"), "_3a_score");
    }

    #[test]
    fn pairs_response_and_points_by_index() {
        let cols = v(&["Daisy ID", "Response 1", "Points 1", "Response 2", "Points 2", "junk"]);
        let d = detect_question_pairs(&cols);
        assert_eq!(d.id_columns, v(&["Daisy ID"]));
        assert_eq!(d.input_columns, v(&["Response 1", "Response 2"]));
        assert_eq!(d.output_columns, v(&["Points 1", "Points 2"]));
    }

    #[test]
    fn unpaired_indices_are_dropped() {
        // Response 3 has no matching Points 3.
        let cols = v(&["id", "Response 1", "Point 1", "Response 3"]);
        let d = detect_question_pairs(&cols);
        assert_eq!(d.id_columns, v(&["id"]));
        assert_eq!(d.input_columns, v(&["Response 1"]));
        assert_eq!(d.output_columns, v(&["Point 1"]));
    }

    #[test]
    fn falls_back_to_first_column_for_id() {
        let cols = v(&["sid", "Response 1", "Points 1"]);
        let d = detect_question_pairs(&cols);
        assert_eq!(d.id_columns, v(&["sid"]));
    }

    #[test]
    fn case_and_spacing_insensitive() {
        let cols = v(&["RESPONSE1", "points1"]);
        let d = detect_question_pairs(&cols);
        assert_eq!(d.input_columns, v(&["RESPONSE1"]));
        assert_eq!(d.output_columns, v(&["points1"]));
    }
}
