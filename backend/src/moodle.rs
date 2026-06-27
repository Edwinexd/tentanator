//! Combine two raw Moodle exports - a *grades* file and a *responses* file -
//! into a single compiled exam sheet with paired `Response N` / `Points N`
//! columns, the shape the rest of Tentanator consumes (see `detect.rs`).
//!
//! Ported from the legacy `combine_moodle_dumps.py`. The join is on `Daisy ID`
//! (robust to differing row order between the two exports); the trailing
//! `Overall average` summary row and zero-max `Q. N /0.00` columns are dropped;
//! `Kommentarer` comment columns carry their `Response Kommentarer` text.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::error::AppResult;
use crate::store;

const DAISY_ID: &str = "Daisy ID";
/// Identifying columns carried into the output (the first three always, the rest
/// only when present in the grades file).
const REQUIRED_INFO: [&str; 3] = [DAISY_ID, "Last name", "First name"];
const OPTIONAL_INFO: [&str; 3] = ["Username", "Email address", "ID number"];

pub struct CombineResult {
    pub headers: Vec<String>,
    pub rows: Vec<Vec<String>>,
    pub students: usize,
    pub questions: usize,
    pub dropped_columns: Vec<String>,
}

struct Pair {
    response_col: String,
    grade_col: Option<String>,
    points_col: Option<String>,
}

fn parse_num(s: &str) -> Option<f64> {
    s.trim().replace(',', ".").parse::<f64>().ok()
}

/// A `Q. X /0.00` column whose max value across all students is 0/blank carries
/// no points and is dropped (matches the legacy `is_zero_max_column`).
fn is_zero_max(header: &str, rows: &[HashMap<String, String>]) -> bool {
    if !header.starts_with("Q. ") || !header.contains("/0.00") {
        return false;
    }
    let max = rows
        .iter()
        .filter_map(|r| r.get(header))
        .filter_map(|v| parse_num(v))
        .fold(None, |acc: Option<f64>, x| Some(acc.map_or(x, |a| a.max(x))));
    max.unwrap_or(0.0) == 0.0
}

/// Combine the two Moodle exports. `grades_path`/`responses_path` point at the
/// raw `.xlsx` files; the result is returned (not written) so the caller picks
/// the destination.
pub fn combine(grades_path: &Path, responses_path: &Path) -> AppResult<CombineResult> {
    let grades_headers = store::get_exam_columns(grades_path)?;
    let mut grades = store::read_exam_data(grades_path)?;
    let response_headers: HashSet<String> =
        store::get_exam_columns(responses_path)?.into_iter().collect();
    let responses = store::read_exam_data(responses_path)?;

    // Drop the trailing "Overall average" summary row Moodle appends.
    if grades
        .last()
        .and_then(|r| r.get("Last name"))
        .map(|v| v.trim() == "Overall average")
        .unwrap_or(false)
    {
        grades.pop();
    }

    // Columns worth zero points are noise; drop them.
    let dropped: Vec<String> = grades_headers
        .iter()
        .filter(|h| is_zero_max(h, &grades))
        .cloned()
        .collect();
    let zero_max: HashSet<&str> = dropped.iter().map(String::as_str).collect();

    // Identifying columns: the three required ones plus any optional present.
    let info_cols: Vec<String> = REQUIRED_INFO
        .iter()
        .chain(OPTIONAL_INFO.iter())
        .filter(|c| grades_headers.iter().any(|h| h == *c))
        .map(|c| c.to_string())
        .collect();

    // Index responses by Daisy ID so order differences don't matter.
    let responses_by_id: HashMap<&str, &HashMap<String, String>> = responses
        .iter()
        .filter_map(|r| r.get(DAISY_ID).map(|d| (d.as_str(), r)))
        .collect();

    // Pair each surviving `Q. N` grade column with its `Response N` column, in
    // grades-file column order.
    let mut pairs: Vec<Pair> = Vec::new();
    for header in &grades_headers {
        if !header.starts_with("Q. ") || zero_max.contains(header.as_str()) {
            continue;
        }
        let q_part = header
            .split('/')
            .next()
            .unwrap_or("")
            .trim_start_matches("Q.")
            .trim()
            .to_string();
        if q_part.contains("Kommentarer") || header.contains("Kommentarer") {
            let suffix = header
                .split("/0.00.")
                .nth(1)
                .map(|n| format!(".{n}"))
                .unwrap_or_default();
            let response_col = format!("Response Kommentarer{suffix}");
            if response_headers.contains(&response_col) {
                pairs.push(Pair { response_col, grade_col: None, points_col: None });
            }
        } else if let Ok(num) = q_part.parse::<i64>() {
            let response_col = format!("Response {num}");
            if response_headers.contains(&response_col) {
                pairs.push(Pair {
                    response_col,
                    grade_col: Some(header.clone()),
                    points_col: Some(format!("Points {num}")),
                });
            }
        }
    }

    // Output headers: id columns, then Response N / Points N for each pair.
    let mut headers = info_cols.clone();
    for p in &pairs {
        headers.push(p.response_col.clone());
        if let Some(points) = &p.points_col {
            headers.push(points.clone());
        }
    }

    // One output row per grades student; responses joined by Daisy ID.
    let mut rows: Vec<Vec<String>> = Vec::with_capacity(grades.len());
    for g in &grades {
        let resp = g.get(DAISY_ID).and_then(|d| responses_by_id.get(d.as_str()));
        let mut rec: Vec<String> = info_cols
            .iter()
            .map(|c| g.get(c).cloned().unwrap_or_default())
            .collect();
        for p in &pairs {
            rec.push(
                resp.and_then(|r| r.get(&p.response_col))
                    .cloned()
                    .unwrap_or_default(),
            );
            if let Some(grade_col) = &p.grade_col {
                rec.push(g.get(grade_col).cloned().unwrap_or_default());
            }
        }
        rows.push(rec);
    }

    let questions = pairs.iter().filter(|p| p.points_col.is_some()).count();
    Ok(CombineResult {
        headers,
        students: rows.len(),
        rows,
        questions,
        dropped_columns: dropped,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn tmp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("tt-moodle-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        dir.join(name)
    }

    #[test]
    fn combines_on_daisy_id_with_pairs() {
        // Grades: two questions (Q.1 worth 1.00, Q.2 worth 0.00 -> dropped) plus
        // the Overall average summary row. Responses are in a different order.
        let grades = tmp("grades.csv");
        std::fs::write(
            &grades,
            "Daisy ID,Last name,First name,Q. 1 /1.00,Q. 2 /0.00\n\
             7,Beta,Bob,2,0\n\
             3,Alpha,Ann,1,0\n\
             ,Overall average,,1.5,0\n",
        )
        .unwrap();
        let responses = tmp("responses.csv");
        std::fs::write(
            &responses,
            "Daisy ID,Response 1,Response 2\n\
             3,ann-answer,x\n\
             7,bob-answer,y\n",
        )
        .unwrap();

        let out = combine(&grades, &responses).unwrap();
        assert_eq!(out.headers, vec!["Daisy ID", "Last name", "First name", "Response 1", "Points 1"]);
        assert_eq!(out.students, 2);
        assert_eq!(out.questions, 1);
        assert_eq!(out.dropped_columns, vec!["Q. 2 /0.00"]);
        // First grades row is Bob (id 7); his response joins from the responses file.
        assert_eq!(out.rows[0], vec!["7", "Beta", "Bob", "bob-answer", "2"]);
        assert_eq!(out.rows[1], vec!["3", "Alpha", "Ann", "ann-answer", "1"]);
    }
}
