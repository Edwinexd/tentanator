//! Second pass: students within a margin of a better grade.
//!
//! The report lists each borderline student with the questions where they
//! still have headroom (points below max), so a reviewer can re-read exactly
//! the answers that could move the grade. The same set round-trips through a
//! review sheet: export it, fill `New points`, import it back. The reviewer is
//! authoritative, so an import overwrites (no conflict queue).

use std::collections::HashMap;

use rust_xlsxwriter::{Format, Workbook};
use serde::Serialize;
use ts_rs::TS;

use crate::domain::{self, Exam, GradedItem};
use crate::error::{AppError, AppResult};
use crate::scheme::{self, GradeGap, LiftSource};

/// Grade source recorded for points changed through a second-pass import.
pub const SOURCE: &str = "second-pass";

const COL_ID: &str = "ID";
const COL_QUESTION: &str = "Question";
const COL_NEW: &str = "New points";
const HEADER: [&str; 11] = [
    COL_ID, "Grade", "Total", "Next grade", "Gap", COL_QUESTION, "Max", "Points",
    "Lifts alone", "Response", COL_NEW,
];

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
    /// Closest to a better grade first.
    pub students: Vec<SecondPassStudent>,
}

/// Every student within `margin` points of a better grade.
pub fn report(exam: &Exam, rows: &[HashMap<String, String>], margin: f64) -> SecondPassReport {
    let Some(scheme) = &exam.scheme else {
        return SecondPassReport { margin, has_scheme: false, students: Vec::new() };
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
    SecondPassReport { margin, has_scheme: true, students }
}

fn xlsx_err(e: rust_xlsxwriter::XlsxError) -> AppError {
    AppError::Other(anyhow::anyhow!(e))
}

/// Review sheet: one row per (borderline student, question with headroom),
/// with an empty `New points` column to fill in and import back.
pub fn build_sheet(exam: &Exam, rep: &SecondPassReport) -> AppResult<(String, Vec<u8>)> {
    let filename = format!("{}_second_pass.xlsx", crate::store::exam_file_stem(exam));
    let mut workbook = Workbook::new();
    let sheet = workbook.add_worksheet();
    let bold = Format::new().set_bold();
    let fill = Format::new().set_background_color("#FFF2CC");
    for (c, h) in HEADER.iter().enumerate() {
        sheet.write_string_with_format(0, c as u16, *h, &bold).map_err(xlsx_err)?;
    }
    let mut r: u32 = 1;
    for st in &rep.students {
        for q in &st.questions {
            sheet.write_string(r, 0, &st.id).map_err(xlsx_err)?;
            sheet.write_string(r, 1, &st.grade).map_err(xlsx_err)?;
            sheet.write_number(r, 2, st.total).map_err(xlsx_err)?;
            sheet.write_string(r, 3, &st.gap.next_grade).map_err(xlsx_err)?;
            sheet.write_number(r, 4, st.gap.points).map_err(xlsx_err)?;
            sheet.write_string(r, 5, &q.col).map_err(xlsx_err)?;
            sheet.write_number(r, 6, q.max).map_err(xlsx_err)?;
            match q.points {
                Some(p) => sheet.write_number(r, 7, p).map_err(xlsx_err)?,
                None => sheet.write_string(r, 7, "ungraded").map_err(xlsx_err)?,
            };
            if let Some(l) = q.lift {
                sheet.write_number(r, 8, l).map_err(xlsx_err)?;
            }
            sheet.write_string(r, 9, &q.response).map_err(xlsx_err)?;
            sheet.write_blank(r, 10, &fill).map_err(xlsx_err)?;
            r += 1;
        }
    }
    for (c, w) in [(0, 10), (5, 12), (8, 11), (9, 60), (10, 11)] {
        sheet.set_column_width(c, w).map_err(xlsx_err)?;
    }
    sheet.set_freeze_panes(1, 0).map_err(xlsx_err)?;
    if r > 1 {
        sheet.autofilter(0, 0, r - 1, HEADER.len() as u16 - 1).map_err(xlsx_err)?;
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
    /// Filled rows whose points equal what is stored.
    pub unchanged: usize,
    /// Rows left blank in `New points`.
    pub blank: usize,
    /// Rows rejected, with the reason (unknown id/question, invalid grade).
    pub invalid: Vec<String>,
    /// Students whose final grade the changes move.
    pub grade_changes: Vec<GradeChange>,
}

/// Plan an import of a filled review sheet against `exam` (no writes). Returns
/// the summary plus the items to store, already applied to the returned exam
/// copy so grade changes are computed from the same data that gets written.
pub fn plan_import(
    exam: &Exam,
    roster: &[HashMap<String, String>],
    sheet: &[HashMap<String, String>],
) -> AppResult<(SecondPassImport, Vec<(String, GradedItem)>)> {
    if let Some(first) = sheet.first() {
        for h in [COL_ID, COL_QUESTION, COL_NEW] {
            if !first.contains_key(h) {
                return Err(AppError::BadRequest(format!(
                    "not a second-pass sheet: missing column `{h}`"
                )));
            }
        }
    }
    let roster_by_id: HashMap<String, &HashMap<String, String>> =
        roster.iter().map(|r| (domain::row_id(r, &exam.id_columns), r)).collect();

    let mut summary = SecondPassImport {
        applied: false,
        changed: Vec::new(),
        unchanged: 0,
        blank: 0,
        invalid: Vec::new(),
        grade_changes: Vec::new(),
    };
    let mut writes: Vec<(String, GradedItem)> = Vec::new();
    for (i, row) in sheet.iter().enumerate() {
        let line = i + 2; // 1-based, after the header
        let cell = |k: &str| row.get(k).map(|v| v.trim().to_string()).unwrap_or_default();
        let (id, col, new) = (cell(COL_ID), cell(COL_QUESTION), cell(COL_NEW));
        if new.is_empty() {
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
        if !before.is_empty() && crate::grade::grades_equal(&before, &new) {
            summary.unchanged += 1;
            continue;
        }
        writes.push((
            col.clone(),
            GradedItem {
                row_id: id.clone(),
                input_text: student.get(&q.input_column).cloned().unwrap_or_default(),
                grade: new.clone(),
                timestamp: crate::store::now_iso(),
            },
        ));
        summary.changed.push(PointChange { id, col, before, after: new });
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
    Ok((summary, writes))
}
