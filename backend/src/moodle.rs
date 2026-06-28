//! Combine two raw Moodle exports - a *grades* file and a *responses* file -
//! into a single compiled exam sheet with paired `Response N` / `Points N`
//! columns, the shape the rest of Tentanator consumes (see `detect.rs`).
//!
//! Ported from the legacy `combine_moodle_dumps.py`. The join is on `Daisy ID`
//! (robust to differing row order between the two exports); the trailing
//! `Overall average` summary row and zero-max `Q. N /0.00` columns are dropped;
//! `Kommentarer` comment columns carry their `Response Kommentarer` text.
//!
//! A student who re-sat after a technical issue appears as several rows with the
//! same `Daisy ID`; those attempts are merged into one row per student, taking
//! the most informative value per cell (see `merge_attempts`). The legacy script
//! left this to a pandas `merge`, which fanned duplicate ids into a cartesian
//! product instead of combining them.

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

/// Moodle's placeholders for "no answer / not yet graded". Treated as empty when
/// merging attempts so a real value from another attempt wins.
fn is_blank(s: &str) -> bool {
    let t = s.trim();
    t.is_empty() || t == "-"
}

/// A `Q. X /0.00` column that actually carries no points is dropped. "No points"
/// means the column is either entirely empty, or fully graded with a numeric max
/// of 0. A column still holding non-numeric tokens (`-`, `Requires grading`) has
/// not been graded yet and is kept - dropping it would silently discard real
/// student responses.
///
/// This mirrors the legacy `is_zero_max_column`, which relied on pandas' string
/// `.max()`: an ungraded column's max is a string (e.g. `-`), which is neither
/// `0` nor NaN, so it was kept. The first Rust port collapsed every non-numeric
/// column to "max 0" and wrongly dropped them (e.g. an all-`-` essay column).
fn is_zero_max(header: &str, rows: &[&HashMap<String, String>]) -> bool {
    if !header.starts_with("Q. ") || !header.contains("/0.00") {
        return false;
    }
    let mut saw_value = false;
    let mut saw_non_numeric = false;
    let mut numeric_max: Option<f64> = None;
    for v in rows.iter().filter_map(|r| r.get(header)) {
        if v.trim().is_empty() {
            continue;
        }
        saw_value = true;
        match parse_num(v) {
            Some(n) => numeric_max = Some(numeric_max.map_or(n, |m: f64| m.max(n))),
            None => saw_non_numeric = true,
        }
    }
    if !saw_value {
        return true; // entirely empty -> NaN max -> drop
    }
    if saw_non_numeric {
        return false; // ungraded (string max) -> keep
    }
    numeric_max.is_none_or(|m| m == 0.0)
}

/// Rank a cell so the most informative attempt wins a merge:
/// 0 = blank, 1 = pending (`Requires grading`), 2 = a real response or grade.
fn cell_rank(s: &str) -> u8 {
    let t = s.trim();
    if is_blank(t) {
        0
    } else if t.eq_ignore_ascii_case("Requires grading") {
        1
    } else {
        2
    }
}

/// Collapse a student's attempts (rows sharing one `Daisy ID`, in file order,
/// which Moodle exports chronologically) into a single row. For each column the
/// highest-ranked value wins; ties resolve to the later attempt, so a redo that
/// rewrote one answer overrides the original while leaving untouched (blank)
/// answers sourced from the earlier attempt.
fn merge_attempts(attempts: &[&HashMap<String, String>]) -> HashMap<String, String> {
    let mut out: HashMap<String, String> = HashMap::new();
    let mut ranks: HashMap<String, u8> = HashMap::new();
    for attempt in attempts {
        for (k, v) in attempt.iter() {
            let r = cell_rank(v);
            match ranks.get(k).copied() {
                // strictly better, or an equal non-blank rank from a later attempt
                Some(prev) if r < prev || (r == prev && r == 0) => {}
                _ => {
                    out.insert(k.clone(), v.clone());
                    ranks.insert(k.clone(), r);
                }
            }
        }
    }
    out
}

/// Moodle appends an `Overall average` summary row (no `Daisy ID`). Detect it by
/// its label so it's dropped even if a future export gives it an id, and so a
/// real student who merely lacks a Daisy ID is not mistaken for it and silently
/// discarded.
fn is_summary_row(r: &HashMap<String, String>) -> bool {
    ["Last name", "First name"].iter().any(|c| {
        r.get(*c)
            .map(|v| v.trim().eq_ignore_ascii_case("Overall average"))
            .unwrap_or(false)
    })
}

/// Group rows by `Daisy ID`, merging each id's attempts into one row. Returns the
/// keys in first-seen order alongside the merged rows. The `Overall average`
/// summary row is dropped; a real student with no `Daisy ID` can't be joined to
/// responses but is kept (under a per-row synthetic key) rather than silently
/// lost.
fn combine_attempts(
    rows: &[HashMap<String, String>],
) -> (Vec<String>, HashMap<String, HashMap<String, String>>) {
    let mut order: Vec<String> = Vec::new();
    let mut groups: HashMap<String, Vec<&HashMap<String, String>>> = HashMap::new();
    for (i, r) in rows.iter().enumerate() {
        if is_summary_row(r) {
            continue;
        }
        let id = r.get(DAISY_ID).map(|s| s.trim()).unwrap_or("");
        // Real IDs are numeric, so the NUL-prefixed synthetic key can't collide
        // with one; it keeps a blank-id student as a distinct (un-joinable) row.
        let key = if id.is_empty() {
            format!("\u{0}noid:{i}")
        } else {
            id.to_string()
        };
        if !groups.contains_key(&key) {
            order.push(key.clone());
        }
        groups.entry(key).or_default().push(r);
    }
    let merged = groups
        .into_iter()
        .map(|(id, attempts)| (id, merge_attempts(&attempts)))
        .collect();
    (order, merged)
}

/// Combine the two Moodle exports. `grades_path`/`responses_path` point at the
/// raw `.xlsx` files; the result is returned (not written) so the caller picks
/// the destination.
pub fn combine(grades_path: &Path, responses_path: &Path) -> AppResult<CombineResult> {
    let grades_headers = store::get_exam_columns(grades_path)?;
    let grades = store::read_exam_data(grades_path)?;
    let response_headers: HashSet<String> =
        store::get_exam_columns(responses_path)?.into_iter().collect();
    let responses = store::read_exam_data(responses_path)?;

    // A student who hit technical issues and re-sat shows up as several rows with
    // the same Daisy ID in both files. Collapse those attempts into one row each
    // (the trailing "Overall average" summary row is dropped by label), then join
    // the two sides by Daisy ID so row order between files is irrelevant.
    let (grade_order, grades_by_id) = combine_attempts(&grades);
    let (_, responses_by_id) = combine_attempts(&responses);
    let merged_grades: Vec<&HashMap<String, String>> = grade_order
        .iter()
        .filter_map(|id| grades_by_id.get(id))
        .collect();

    // Columns worth zero points are noise; drop them.
    let dropped: Vec<String> = grades_headers
        .iter()
        .filter(|h| is_zero_max(h, &merged_grades))
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

    // One output row per unique student (attempts already merged), in the order
    // they first appear in the grades file; responses joined by Daisy ID.
    let mut rows: Vec<Vec<String>> = Vec::with_capacity(grade_order.len());
    for id in &grade_order {
        let g = &grades_by_id[id];
        let resp = responses_by_id.get(id);
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

    #[test]
    fn keeps_student_without_daisy_id_drops_summary() {
        // Gabe has no Daisy ID (so no response can join), but his grade must not be
        // dropped; only the labelled "Overall average" row is removed.
        let grades = tmp("grades-noid.csv");
        std::fs::write(
            &grades,
            "Daisy ID,Last name,First name,Q. 1 /1.00\n\
             7,Beta,Bob,2\n\
             ,Gamma,Gabe,1\n\
             ,Overall average,,1.5\n",
        )
        .unwrap();
        let responses = tmp("responses-noid.csv");
        std::fs::write(&responses, "Daisy ID,Response 1\n7,bob-answer\n").unwrap();

        let out = combine(&grades, &responses).unwrap();
        assert_eq!(out.students, 2, "blank-id student kept, summary row dropped");
        assert_eq!(out.rows[0], vec!["7", "Beta", "Bob", "bob-answer", "2"]);
        assert_eq!(out.rows[1], vec!["", "Gamma", "Gabe", "", "1"]);
    }

    #[test]
    fn merges_multiple_attempts_into_one_row() {
        // Jane (id 5) re-sat after a technical issue: the second attempt only
        // redid Q2 and left everything else blank ("-"). Both files carry her
        // twice. Expect one combined row, with Q2 taking the later (redone)
        // answer and Q1 falling back to the first attempt.
        let grades = tmp("grades-attempts.csv");
        std::fs::write(
            &grades,
            "Daisy ID,Last name,First name,Q. 1 /1.00,Q. 2 /1.00\n\
             5,Doe,Jane,Requires grading,Requires grading\n\
             5,Doe,Jane,-,Requires grading\n\
             9,Roe,Rick,2,3\n",
        )
        .unwrap();
        let responses = tmp("responses-attempts.csv");
        std::fs::write(
            &responses,
            "Daisy ID,Response 1,Response 2\n\
             5,first-q1,first-q2\n\
             5,-,redone-q2\n\
             9,rick-q1,rick-q2\n",
        )
        .unwrap();

        let out = combine(&grades, &responses).unwrap();
        assert_eq!(out.students, 2, "two attempts for id 5 collapse to one row");
        assert_eq!(out.questions, 2);
        assert!(out.dropped_columns.is_empty());
        // Q1 keeps the first attempt's answer/grade (second was blank); Q2 takes
        // the later redo; both attempts agree the grade is still pending.
        assert_eq!(
            out.rows[0],
            vec!["5", "Doe", "Jane", "first-q1", "Requires grading", "redone-q2", "Requires grading"]
        );
        assert_eq!(out.rows[1], vec!["9", "Roe", "Rick", "rick-q1", "2", "rick-q2", "3"]);
    }

    #[test]
    fn keeps_ungraded_zero_max_columns() {
        // Three /0.00 columns: Q2 still ungraded ("-"/"Requires grading") must be
        // kept (it holds real responses); Q3 graded numerically to 0 is dropped;
        // Q4 entirely empty is dropped.
        let grades = tmp("grades-zeromax.csv");
        std::fs::write(
            &grades,
            "Daisy ID,Last name,First name,Q. 1 /1.00,Q. 2 /0.00,Q. 3 /0.00,Q. 4 /0.00\n\
             1,A,Aa,1,-,0,\n\
             2,B,Bb,1,Requires grading,0,\n",
        )
        .unwrap();
        let responses = tmp("responses-zeromax.csv");
        std::fs::write(
            &responses,
            "Daisy ID,Response 1,Response 2,Response 3,Response 4\n\
             1,r1-a,r2-a,r3-a,r4-a\n\
             2,r1-b,r2-b,r3-b,r4-b\n",
        )
        .unwrap();

        let out = combine(&grades, &responses).unwrap();
        assert_eq!(out.dropped_columns, vec!["Q. 3 /0.00", "Q. 4 /0.00"]);
        assert!(out.headers.contains(&"Response 2".to_string()));
        assert!(out.headers.contains(&"Points 2".to_string()));
        assert!(!out.headers.contains(&"Response 3".to_string()));
        assert!(!out.headers.contains(&"Response 4".to_string()));
    }
}
