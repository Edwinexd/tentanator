//! Second pass: students within a margin of a better grade.
//!
//! The report lists each borderline student with the questions where they
//! still have headroom (points below max), so a reviewer can re-read exactly
//! the answers that could move the grade. The same set round-trips through a
//! review sheet: one row per student, edit the points in place, import it
//! back. The trailing state column says what every points cell held at export,
//! so only edited cells count. The reviewer is authoritative, so an import
//! overwrites, except when the stored points also moved since export: that is
//! a genuine conflict and goes to the conflict queue instead.

use std::collections::{BTreeMap, HashMap};

use rust_xlsxwriter::{Format, Workbook};
use serde::Serialize;
use ts_rs::TS;

use crate::domain::{self, Exam, GradedItem};
use crate::error::{AppError, AppResult};
use crate::scheme::{self, GradeGap, LiftSource, SchemeRounding};
use crate::snapshot::{self, Merge, RowState, COL_STATE};

/// Grade source recorded for points changed through a second-pass import.
pub const SOURCE: &str = "second-pass";

const COL_ID: &str = "ID";
const HEADER: [&str; 5] = [COL_ID, "Grade", "Total", "Next grade", "Gap"];
// Legacy long format (one row per student and question, `New points` to fill).
const COL_QUESTION: &str = "Question";
const COL_POINTS: &str = "Points";
const COL_NEW: &str = "New points";
const UNGRADED: &str = "ungraded";

/// A question where a borderline student still has headroom.
#[derive(Clone, Debug, Serialize, TS)]
#[ts(export, export_to = "../../web/src/lib/generated/")]
pub struct SecondPassQuestion {
    /// Output column (what the grade route and the review sheet key on).
    pub col: String,
    pub label: String,
    pub group: String,
    pub response: String,
    /// Current points (`None` = ungraded).
    pub points: Option<f64>,
    pub max: f64,
    /// Extra points on this question alone that reach a better grade.
    pub lift: Option<f64>,
}

#[derive(Clone, Debug, Serialize, TS)]
#[ts(export, export_to = "../../web/src/lib/generated/")]
pub struct SecondPassStudent {
    pub id: String,
    pub grade: String,
    pub total: f64,
    pub complete: bool,
    /// Lifts keyed by group tag or output column (not scheme var).
    pub gap: GradeGap,
    pub questions: Vec<SecondPassQuestion>,
}

#[derive(Clone, Debug, Serialize, TS)]
#[ts(export, export_to = "../../web/src/lib/generated/")]
pub struct SecondPassReport {
    pub margin: f64,
    pub has_scheme: bool,
    /// The scheme's rounding, when on: grades are already rounded and the
    /// margin counts graded points beyond it.
    pub rounding: Option<SchemeRounding>,
    /// Closest to a better grade first.
    pub students: Vec<SecondPassStudent>,
}

/// Every student within `margin` points of a better grade.
pub fn report(exam: &Exam, rows: &[HashMap<String, String>], margin: f64) -> SecondPassReport {
    let Some(scheme) = &exam.scheme else {
        return SecondPassReport { margin, has_scheme: false, rounding: None, students: Vec::new() };
    };
    let questions = exam.scheme_questions();
    let col_of: HashMap<String, &String> =
        exam.configured_questions().map(|(col, q)| (q.effective_var(col), col)).collect();

    let mut students = Vec::new();
    for row in rows {
        let rid = domain::row_id(row, &exam.id_columns);
        let points = exam.student_points(&rid);
        let Some(mut gap) = scheme::grade_gap(scheme, &questions, &points, margin) else {
            continue;
        };
        for l in &mut gap.lifts {
            if l.source == LiftSource::Question {
                if let Some(col) = col_of.get(&l.key) {
                    l.key = col.to_string();
                }
            }
        }
        let sr = scheme::compute_student(&rid, scheme, &questions, &points);
        let qs = exam
            .configured_questions()
            .filter(|(_, q)| q.max_points > 0.0)
            .filter_map(|(col, q)| {
                let pts = q.points_for(&rid);
                if pts.unwrap_or(0.0) >= q.max_points {
                    return None;
                }
                let lift = gap
                    .lifts
                    .iter()
                    .find(|l| l.source == LiftSource::Question && &l.key == col)
                    .map(|l| l.points);
                Some(SecondPassQuestion {
                    col: col.clone(),
                    label: q.label(col),
                    group: q.group.clone(),
                    response: row.get(&q.input_column).cloned().unwrap_or_default(),
                    points: pts,
                    max: q.max_points,
                    lift,
                })
            })
            .collect();
        students.push(SecondPassStudent {
            id: rid,
            grade: sr.grade,
            total: sr.total,
            complete: sr.complete,
            gap,
            questions: qs,
        });
    }
    students.sort_by(|a, b| {
        a.gap
            .points
            .partial_cmp(&b.gap.points)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.id.cmp(&b.id))
    });
    SecondPassReport { margin, has_scheme: true, rounding: scheme::active_rounding(scheme).cloned(), students }
}

fn xlsx_err(e: rust_xlsxwriter::XlsxError) -> AppError {
    AppError::Other(anyhow::anyhow!(e))
}

/// Review sheet: one row per borderline student, then a response and a points
/// column (headed by the exam's own input/output column names) for every
/// question where some listed student has headroom. Points cells with headroom
/// are highlighted; the reviewer edits points in place.
pub fn build_sheet(
    exam: &Exam,
    roster: &[HashMap<String, String>],
    rep: &SecondPassReport,
) -> AppResult<(String, Vec<u8>)> {
    let filename = format!("{}_second_pass.xlsx", crate::store::exam_file_stem(exam));
    let roster_by_id: HashMap<String, &HashMap<String, String>> =
        roster.iter().map(|r| (domain::row_id(r, &exam.id_columns), r)).collect();
    let questions: Vec<(&String, &domain::QuestionGrades)> = exam
        .configured_questions()
        .filter(|(col, _)| rep.students.iter().any(|st| st.questions.iter().any(|q| &q.col == *col)))
        .collect();

    let mut workbook = Workbook::new();
    let sheet = workbook.add_worksheet();
    let bold = Format::new().set_bold();
    let fill = Format::new().set_background_color("#FFF2CC");
    let headers = HEADER
        .iter()
        .map(|h| h.to_string())
        .chain(questions.iter().flat_map(|(col, q)| [q.input_column.clone(), (*col).clone()]))
        .chain([COL_STATE.to_string()]);
    for (c, h) in headers.enumerate() {
        sheet.write_string_with_format(0, c as u16, &h, &bold).map_err(xlsx_err)?;
    }
    let state_col = (HEADER.len() + 2 * questions.len()) as u16;
    for (i, st) in rep.students.iter().enumerate() {
        let r = i as u32 + 1;
        sheet.write_string(r, 0, &st.id).map_err(xlsx_err)?;
        sheet.write_string(r, 1, &st.grade).map_err(xlsx_err)?;
        sheet.write_number(r, 2, st.total).map_err(xlsx_err)?;
        sheet.write_string(r, 3, &st.gap.next_grade).map_err(xlsx_err)?;
        sheet.write_number(r, 4, st.gap.points).map_err(xlsx_err)?;
        let student = roster_by_id.get(&st.id);
        let mut state = BTreeMap::new();
        for (j, (col, q)) in questions.iter().enumerate() {
            let c = (HEADER.len() + 2 * j) as u16;
            let response = student.and_then(|r| r.get(&q.input_column)).cloned().unwrap_or_default();
            sheet.write_string(r, c, &response).map_err(xlsx_err)?;
            let headroom = st.questions.iter().any(|sq| &sq.col == *col);
            let points = q.points_for(&st.id);
            match (points, headroom) {
                (Some(p), true) => {
                    sheet.write_number_with_format(r, c + 1, p, &fill).map_err(xlsx_err)?;
                }
                (Some(p), false) => {
                    sheet.write_number(r, c + 1, p).map_err(xlsx_err)?;
                }
                (None, true) => {
                    sheet.write_blank(r, c + 1, &fill).map_err(xlsx_err)?;
                }
                (None, false) => {}
            }
            state.insert((*col).clone(), points.map(|p| p.to_string()).unwrap_or_default());
        }
        sheet
            .write_string(r, state_col, RowState::new(&exam.name, &st.id, state).encode())
            .map_err(xlsx_err)?;
    }
    for j in 0..questions.len() {
        let c = (HEADER.len() + 2 * j) as u16;
        sheet.set_column_width(c, 30).map_err(xlsx_err)?;
        sheet.set_column_width(c + 1, 10).map_err(xlsx_err)?;
    }
    sheet.set_column_width(state_col, 12).map_err(xlsx_err)?;
    sheet.set_freeze_panes(1, 1).map_err(xlsx_err)?;
    if !rep.students.is_empty() {
        sheet
            .autofilter(0, 0, rep.students.len() as u32, state_col)
            .map_err(xlsx_err)?;
    }
    let bytes = workbook.save_to_buffer().map_err(xlsx_err)?;
    Ok((filename, bytes))
}

#[derive(Clone, Debug, Serialize, TS)]
#[ts(export, export_to = "../../web/src/lib/generated/")]
pub struct PointChange {
    pub id: String,
    pub col: String,
    /// Previous stored grade (empty when there was none).
    pub before: String,
    pub after: String,
}

#[derive(Clone, Debug, Serialize, TS)]
#[ts(export, export_to = "../../web/src/lib/generated/")]
pub struct GradeChange {
    pub id: String,
    pub before: String,
    pub after: String,
}

#[derive(Clone, Debug, Serialize, TS)]
#[ts(export, export_to = "../../web/src/lib/generated/")]
pub struct SecondPassImport {
    /// False for a dry run (nothing written).
    pub applied: bool,
    pub changed: Vec<PointChange>,
    /// Points changed in the sheet whose stored value also moved since export
    /// (`before` = stored now). Queued as grade conflicts, not written.
    pub conflicts: Vec<PointChange>,
    /// Edited cells whose points equal what is stored.
    pub unchanged: usize,
    /// Cells left as exported (or `New points` left blank in an old sheet).
    pub blank: usize,
    /// Cells rejected, with the reason (unknown id/question, invalid grade).
    pub invalid: Vec<String>,
    /// Students whose final grade the changes move.
    pub grade_changes: Vec<GradeChange>,
}

/// Items to store and items to queue as conflicts, keyed by output column.
pub type PlannedItems = Vec<(String, GradedItem)>;

/// One reviewed cell: the sheet's value and, with state, the value at export.
struct Entry {
    line: usize,
    id: String,
    col: String,
    new: String,
    base: Option<String>,
}

/// Cells of a current sheet (one row per student, points edited in place).
fn wide_entries(exam: &Exam, sheet: &[HashMap<String, String>]) -> (Vec<Entry>, Vec<String>) {
    let mut entries = Vec::new();
    let mut invalid = Vec::new();
    for (i, row) in sheet.iter().enumerate() {
        let line = i + 2; // 1-based, after the header
        let id = row.get(COL_ID).map(|v| v.trim().to_string()).unwrap_or_default();
        let Some(state) = RowState::of_row(row, &exam.name).filter(|st| st.id == id) else {
            if !id.is_empty() {
                invalid.push(format!("row {line}: `{COL_STATE}` missing or not for `{id}`"));
            }
            continue;
        };
        for (col, base) in state.cells {
            let new = row.get(&col).map(|v| v.trim().to_string()).unwrap_or_default();
            entries.push(Entry { line, id: id.clone(), col, new, base: Some(base) });
        }
    }
    (entries, invalid)
}

/// Cells of an old long sheet (one row per student and question, `New
/// points`). With state, an edited `Points` cell counts when `New points` is
/// blank.
fn long_entries(exam: &Exam, sheet: &[HashMap<String, String>]) -> Vec<Entry> {
    sheet
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let cell = |k: &str| row.get(k).map(|v| v.trim().to_string()).unwrap_or_default();
            let (id, col, mut new) = (cell(COL_ID), cell(COL_QUESTION), cell(COL_NEW));
            let base = RowState::of_row(row, &exam.name)
                .filter(|st| st.id == id)
                .and_then(|st| st.base(&col).map(str::to_string));
            if new.is_empty() && base.is_some() {
                let points = cell(COL_POINTS);
                new = if points == UNGRADED { String::new() } else { points };
            }
            Entry { line: i + 2, id, col, new, base }
        })
        .collect()
}

/// Plan an import of a filled review sheet against `exam` (no writes). Returns
/// the summary, the items to store and the conflicting items to queue; grade
/// changes are computed from exactly the items that get written. Accepts the
/// current one-row-per-student sheet and the old long one.
pub fn plan_import(
    exam: &Exam,
    roster: &[HashMap<String, String>],
    sheet: &[HashMap<String, String>],
) -> AppResult<(SecondPassImport, PlannedItems, PlannedItems)> {
    let first = sheet.first();
    let has = |h: &str| first.is_some_and(|r| r.contains_key(h));
    let mut summary = SecondPassImport {
        applied: false,
        changed: Vec::new(),
        conflicts: Vec::new(),
        unchanged: 0,
        blank: 0,
        invalid: Vec::new(),
        grade_changes: Vec::new(),
    };
    let entries = if has(COL_QUESTION) && has(COL_NEW) {
        long_entries(exam, sheet)
    } else if first.is_none() || (has(COL_ID) && has(COL_STATE)) {
        let (entries, invalid) = wide_entries(exam, sheet);
        summary.invalid = invalid;
        entries
    } else {
        return Err(AppError::BadRequest(format!(
            "not a second-pass sheet: needs `{COL_ID}` and `{COL_STATE}` columns"
        )));
    };
    let roster_by_id: HashMap<String, &HashMap<String, String>> =
        roster.iter().map(|r| (domain::row_id(r, &exam.id_columns), r)).collect();

    let mut writes: PlannedItems = Vec::new();
    let mut conflicts: PlannedItems = Vec::new();
    for Entry { line, id, col, new, base } in entries {
        let untouched = base.as_deref().is_some_and(|b| crate::grade::grades_equal(b, &new));
        if new.is_empty() || untouched {
            summary.blank += 1;
            continue;
        }
        let Some(student) = roster_by_id.get(&id) else {
            summary.invalid.push(format!("row {line}: unknown ID `{id}`"));
            continue;
        };
        let Some(q) = exam.questions.get(&col) else {
            summary.invalid.push(format!("row {line}: unknown question `{col}`"));
            continue;
        };
        if let Err(e) = crate::grade::validate_grade(&new) {
            summary.invalid.push(format!("row {line}: {col} `{new}`: {e}"));
            continue;
        }
        let before = q
            .graded_items
            .iter()
            .find(|gi| gi.row_id == id)
            .map(|gi| gi.grade.clone())
            .unwrap_or_default();
        let item = GradedItem {
            row_id: id.clone(),
            input_text: student.get(&q.input_column).cloned().unwrap_or_default(),
            grade: new.clone(),
            timestamp: crate::store::now_iso(),
        };
        let change = PointChange { id, col: col.clone(), before: before.clone(), after: new.clone() };
        match snapshot::merge(base.as_deref(), &before, &new) {
            Merge::Same => summary.unchanged += 1,
            // Without state the reviewer wins, as before the state column existed.
            Merge::Conflict if base.is_some() => {
                conflicts.push((col, item));
                summary.conflicts.push(change);
            }
            _ => {
                writes.push((col, item));
                summary.changed.push(change);
            }
        }
    }

    if let Some(scheme) = &exam.scheme {
        let mut after = exam.clone();
        for (col, item) in &writes {
            if let Some(q) = after.questions.get_mut(col) {
                q.graded_items.retain(|gi| gi.row_id != item.row_id);
                q.graded_items.push(item.clone());
            }
        }
        let questions = exam.scheme_questions();
        let mut ids: Vec<&String> = writes.iter().map(|(_, it)| &it.row_id).collect();
        ids.sort();
        ids.dedup();
        for id in ids {
            let was = scheme::compute_student(id, scheme, &questions, &exam.student_points(id));
            let now = scheme::compute_student(id, scheme, &questions, &after.student_points(id));
            if was.grade != now.grade {
                summary.grade_changes.push(GradeChange {
                    id: id.clone(),
                    before: was.grade,
                    after: now.grade,
                });
            }
        }
    }
    Ok((summary, writes, conflicts))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exam_with(grades: &[(&str, &str)]) -> Exam {
        let mut exam = Exam {
            name: "ex".into(),
            exam_file: "e.xlsx".into(),
            id_columns: vec!["id".into()],
            input_columns: vec!["ans".into()],
            output_columns: vec!["Q1".into()],
            course: None,
            last_updated: String::new(),
            questions: HashMap::new(),
            scheme: None,
        };
        let q = exam.ensure_question("Q1");
        for (id, g) in grades {
            q.graded_items.push(GradedItem {
                row_id: (*id).into(),
                input_text: String::new(),
                grade: (*g).into(),
                timestamp: String::new(),
            });
        }
        exam
    }

    fn row(id: &str, points: &str, new: &str, base: Option<&str>) -> HashMap<String, String> {
        let mut r = HashMap::from([
            (COL_ID.to_string(), id.to_string()),
            (COL_QUESTION.to_string(), "Q1".to_string()),
            (COL_POINTS.to_string(), points.to_string()),
            (COL_NEW.to_string(), new.to_string()),
        ]);
        if let Some(b) = base {
            let st = RowState::new("ex", id, BTreeMap::from([("Q1".into(), b.into())]));
            r.insert(COL_STATE.into(), st.encode());
        }
        r
    }

    #[test]
    fn state_column_separates_changes_from_conflicts() {
        // s1 moved 2 -> 3 since export; s2 and s3 are unchanged at 2.
        let exam = exam_with(&[("s1", "3"), ("s2", "2"), ("s3", "2"), ("s4", "3")]);
        let roster: Vec<_> = ["s1", "s2", "s3", "s4"]
            .iter()
            .map(|id| HashMap::from([("id".to_string(), id.to_string())]))
            .collect();
        let sheet = vec![
            row("s1", "2", "4", Some("2")), // both sides changed: conflict
            row("s2", "2", "4", Some("2")), // only the sheet changed: write
            row("s3", "5", "", Some("2")),  // `Points` edited in place: write
            row("s4", "2", "4", None),      // old sheet: reviewer wins
        ];
        let (summary, writes, conflicts) = plan_import(&exam, &roster, &sheet).unwrap();
        let ids = |v: &PlannedItems| v.iter().map(|(_, it)| it.row_id.clone()).collect::<Vec<_>>();
        assert_eq!(ids(&writes), ["s2", "s3", "s4"]);
        assert_eq!(writes[1].1.grade, "5");
        assert_eq!(ids(&conflicts), ["s1"]);
        assert_eq!(summary.conflicts[0].before, "3");
    }

    #[test]
    fn wide_sheet_imports_only_edited_cells() {
        let exam = exam_with(&[("s1", "3"), ("s2", "2")]);
        let roster: Vec<_> = ["s1", "s2"]
            .iter()
            .map(|id| HashMap::from([("id".to_string(), id.to_string())]))
            .collect();
        let wide = |id: &str, pts: &str, base: &str| {
            let st = RowState::new("ex", id, BTreeMap::from([("Q1".into(), base.into())]));
            HashMap::from([
                (COL_ID.to_string(), id.to_string()),
                ("Q1".to_string(), pts.to_string()),
                (COL_STATE.to_string(), st.encode()),
            ])
        };
        let sheet = vec![
            wide("s1", "2", "2"), // untouched although the store moved to 3
            wide("s2", "4", "2"), // edited in place
        ];
        let (summary, writes, conflicts) = plan_import(&exam, &roster, &sheet).unwrap();
        assert_eq!(summary.blank, 1);
        assert_eq!(writes.len(), 1);
        assert_eq!((writes[0].1.row_id.as_str(), writes[0].1.grade.as_str()), ("s2", "4"));
        assert!(conflicts.is_empty());
        let foreign = vec![HashMap::from([("x".to_string(), "1".to_string())])];
        assert!(plan_import(&exam, &roster, &foreign).is_err());
    }
}
