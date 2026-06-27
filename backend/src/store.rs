//! Persistence. Exam state lives in the Turso database (`db.rs`); exam files and
//! exported Excel stay on disk.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use calamine::{open_workbook_auto, Data, Reader};
use serde_json::Value;
use turso::Connection;

use crate::config::Config;
use crate::db;
use crate::domain::{
    BankQuestion, Exam, ExamSummary, GradeConflict, GradedItem, QuestionGrades, Session,
    SessionSummary,
};
use crate::error::{AppError, AppResult};

// ---------------------------------------------------------------------------
// Timestamps & name sanitization
// ---------------------------------------------------------------------------

pub fn now_iso() -> String {
    chrono::Local::now().format("%Y-%m-%dT%H:%M:%S%.6f").to_string()
}

pub fn timestamp_compact() -> String {
    chrono::Local::now().format("%Y%m%d_%H%M%S").to_string()
}

pub fn sanitize_name(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
        .collect()
}

fn json_array(v: &[String]) -> String {
    serde_json::to_string(v).unwrap_or_else(|_| "[]".to_string())
}

fn parse_array(s: &str) -> Vec<String> {
    serde_json::from_str(s).unwrap_or_default()
}

fn opt(s: String) -> Option<String> {
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

// ---------------------------------------------------------------------------
// Exam files (on disk)
// ---------------------------------------------------------------------------

pub fn list_exam_files(config: &Config) -> Vec<String> {
    let dir = config.exams_dir();
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for e in entries.flatten() {
            if let Some(name) = e.file_name().to_str() {
                let lower = name.to_lowercase();
                if lower.ends_with(".xlsx") || lower.ends_with(".xls") || lower.ends_with(".csv") {
                    out.push(name.to_string());
                }
            }
        }
    }
    out.sort();
    out
}

/// List scanned exam PDFs available for results rendering (scans/ + exams/).
pub fn list_pdf_files(config: &Config) -> Vec<String> {
    let mut out = Vec::new();
    for dir in [config.data_dir.join("scans"), config.exams_dir()] {
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for e in entries.flatten() {
                if let Some(n) = e.file_name().to_str() {
                    if n.to_lowercase().ends_with(".pdf") {
                        out.push(n.to_string());
                    }
                }
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

pub fn resolve_exam_path(config: &Config, exam_file: &str) -> Option<PathBuf> {
    let direct = config.exams_dir().join(exam_file);
    if direct.exists() {
        return Some(direct);
    }
    let stem = Path::new(exam_file)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(exam_file);
    for ext in ["xlsx", "xls", "csv"] {
        let alt = config.exams_dir().join(format!("{stem}.{ext}"));
        if alt.exists() {
            return Some(alt);
        }
    }
    None
}

fn cell_to_string(c: &Data) -> String {
    match c {
        Data::Empty => String::new(),
        Data::String(s) => s.clone(),
        Data::Int(i) => i.to_string(),
        Data::Float(f) => {
            if f.fract() == 0.0 && f.is_finite() {
                format!("{}", *f as i64)
            } else {
                f.to_string()
            }
        }
        Data::Bool(b) => b.to_string(),
        other => other.to_string(),
    }
}

enum SheetFmt {
    /// xlsx / xls(binary OLE) / ods - handled by calamine.
    Calamine,
    /// SpreadsheetML 2003 XML (`.xls` that is really `<?mso-application?>` XML,
    /// as produced by Daisy/Excel "XML Spreadsheet" export). calamine can't read
    /// this, so it has its own parser.
    SpreadsheetXml,
    /// Plain CSV / TSV.
    Csv,
}

/// Pick the parser from the file's actual content, not its extension - exports
/// named `.xls` are often XML or CSV rather than the binary OLE format.
fn sniff_format(path: &Path) -> SheetFmt {
    use std::io::Read;
    let mut buf = [0u8; 4096];
    let n = std::fs::File::open(path)
        .and_then(|mut f| f.read(&mut buf))
        .unwrap_or(0);
    let head = &buf[..n];
    if head.starts_with(b"PK\x03\x04") || head.starts_with(&[0xD0, 0xCF, 0x11, 0xE0]) {
        return SheetFmt::Calamine;
    }
    let text = String::from_utf8_lossy(head);
    if text.contains("urn:schemas-microsoft-com:office:spreadsheet")
        || text.contains("mso-application")
    {
        return SheetFmt::SpreadsheetXml;
    }
    SheetFmt::Csv
}

/// Parse a SpreadsheetML 2003 XML workbook into rows of strings (first row =
/// headers). Honours `ss:Index` for sparse cells; entities are decoded by the
/// XML parser.
fn parse_spreadsheet_xml(text: &str) -> AppResult<Vec<Vec<String>>> {
    let doc = roxmltree::Document::parse(text)
        .map_err(|e| AppError::BadRequest(format!("cannot parse XML spreadsheet: {e}")))?;
    let Some(table) = doc.descendants().find(|n| n.tag_name().name() == "Table") else {
        return Ok(Vec::new());
    };
    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut maxcols = 0usize;
    for row in table.children().filter(|n| n.tag_name().name() == "Row") {
        let mut cells: Vec<String> = Vec::new();
        let mut col = 0usize; // next 0-based column to write
        for cell in row.children().filter(|n| n.tag_name().name() == "Cell") {
            if let Some(idx) = cell
                .attributes()
                .find(|a| a.name() == "Index")
                .and_then(|a| a.value().parse::<usize>().ok())
            {
                col = idx.saturating_sub(1);
            }
            while cells.len() < col {
                cells.push(String::new());
            }
            let val = cell
                .children()
                .find(|n| n.tag_name().name() == "Data")
                .and_then(|d| d.text())
                .unwrap_or("")
                .to_string();
            if cells.len() == col {
                cells.push(val);
            } else {
                cells[col] = val;
            }
            col += 1;
        }
        maxcols = maxcols.max(cells.len());
        rows.push(cells);
    }
    for r in &mut rows {
        while r.len() < maxcols {
            r.push(String::new());
        }
    }
    Ok(rows)
}

fn calamine_rows(path: &Path) -> AppResult<Vec<Vec<String>>> {
    let mut wb = open_workbook_auto(path)
        .map_err(|e| AppError::BadRequest(format!("cannot open workbook: {e}")))?;
    let range = wb
        .worksheet_range_at(0)
        .ok_or_else(|| AppError::BadRequest("workbook has no sheets".into()))?
        .map_err(|e| AppError::BadRequest(format!("cannot read sheet: {e}")))?;
    Ok(range
        .rows()
        .map(|r| r.iter().map(cell_to_string).collect())
        .collect())
}

fn csv_rows(path: &Path) -> AppResult<Vec<Vec<String>>> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .flexible(true)
        .from_path(path)?;
    let mut out = Vec::new();
    for rec in rdr.records() {
        out.push(rec?.iter().map(|s| s.to_string()).collect());
    }
    Ok(out)
}

/// All rows (header + data) of an exam file, format auto-detected.
fn sheet_rows(path: &Path) -> AppResult<Vec<Vec<String>>> {
    match sniff_format(path) {
        SheetFmt::Calamine => calamine_rows(path),
        SheetFmt::SpreadsheetXml => parse_spreadsheet_xml(&std::fs::read_to_string(path)?),
        SheetFmt::Csv => csv_rows(path),
    }
}

pub fn get_exam_columns(path: &Path) -> AppResult<Vec<String>> {
    Ok(sheet_rows(path)?.into_iter().next().unwrap_or_default())
}

pub fn read_exam_data(path: &Path) -> AppResult<Vec<HashMap<String, String>>> {
    let mut rows = sheet_rows(path)?.into_iter();
    let headers = rows.next().unwrap_or_default();
    let mut out = Vec::new();
    for rec in rows {
        let mut map = HashMap::new();
        for (j, h) in headers.iter().enumerate() {
            map.insert(h.clone(), rec.get(j).cloned().unwrap_or_default());
        }
        out.push(map);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Exams (Turso)
// ---------------------------------------------------------------------------

pub async fn exam_exists(conn: &Connection, name: &str) -> AppResult<bool> {
    let mut r = conn.query("SELECT 1 FROM exams WHERE name = ?", (name,)).await?;
    Ok(r.next().await?.is_some())
}

pub async fn insert_exam(conn: &Connection, exam: &Exam) -> AppResult<()> {
    conn.execute("BEGIN", ()).await?;
    let res = insert_exam_inner(conn, exam).await;
    if res.is_ok() {
        conn.execute("COMMIT", ()).await?;
    } else {
        let _ = conn.execute("ROLLBACK", ()).await;
    }
    res
}

async fn insert_exam_inner(conn: &Connection, e: &Exam) -> AppResult<()> {
    conn.execute("DELETE FROM exams WHERE name = ?", (e.name.as_str(),)).await?;
    let scheme = e
        .scheme
        .as_ref()
        .map(|sc| serde_json::to_string(sc).unwrap_or_default())
        .unwrap_or_default();
    conn.execute(
        "INSERT INTO exams \
         (name, exam_file, id_columns, input_columns, output_columns, course, last_updated, scheme, archived) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)",
        (
            e.name.as_str(),
            e.exam_file.as_str(),
            json_array(&e.id_columns),
            json_array(&e.input_columns),
            json_array(&e.output_columns),
            e.course.clone().unwrap_or_default(),
            e.last_updated.as_str(),
            scheme,
        ),
    )
    .await?;
    for (col, q) in &e.questions {
        upsert_question_row(conn, &e.name, col, q).await?;
        // Grades carried in on the Exam (legacy import) land in the default session.
        for item in &q.graded_items {
            put_graded_item(conn, &e.name, col, item, "", DEFAULT_SESSION).await?;
        }
    }
    Ok(())
}

const DEFAULT_SESSION: &str = "default";

pub async fn upsert_question_row(
    conn: &Connection,
    exam: &str,
    col: &str,
    q: &QuestionGrades,
) -> AppResult<()> {
    conn.execute(
        "DELETE FROM questions WHERE exam = ? AND output_col = ?",
        (exam, col),
    )
    .await?;
    let sr = q
        .sampling_result
        .as_ref()
        .map(|r| serde_json::to_string(r).unwrap_or_default())
        .unwrap_or_default();
    conn.execute(
        "INSERT INTO questions \
         (exam, output_col, question_name, input_column, exam_question, sample_answer, global_question_id, sampling_result, var, qgroup, qtype, max_points, position, estimate) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            exam,
            col,
            q.question_name.as_str(),
            q.input_column.as_str(),
            q.exam_question.as_str(),
            q.sample_answer.as_str(),
            q.global_question_id.clone().unwrap_or_default(),
            sr,
            q.var.as_str(),
            q.group.as_str(),
            q.qtype.as_str(),
            q.max_points,
            q.position as i64,
            q.estimate.clone().unwrap_or_default(),
        ),
    )
    .await?;
    Ok(())
}

pub async fn put_graded_item(
    conn: &Connection,
    exam: &str,
    col: &str,
    item: &GradedItem,
    source: &str,
    session: &str,
) -> AppResult<()> {
    conn.execute(
        "DELETE FROM graded_items WHERE exam = ? AND output_col = ? AND row_id = ?",
        (exam, col, item.row_id.as_str()),
    )
    .await?;
    conn.execute(
        "INSERT INTO graded_items (exam, output_col, row_id, input_text, grade, timestamp, source, session) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            exam,
            col,
            item.row_id.as_str(),
            item.input_text.as_str(),
            item.grade.as_str(),
            item.timestamp.as_str(),
            source,
            session,
        ),
    )
    .await?;
    // Mirror into the cross-exam pool if the question is linked.
    if let Some(gq) = question_gq_id(conn, exam, col).await? {
        sync_item_to_pool(conn, &gq, exam, item).await?;
    }
    Ok(())
}

/// Interpret a pre-filled output-cell value as a grade. A numeric value is taken
/// as-is; a dash (`-`) means no points awarded and is treated as `0` (so
/// comment/feedback questions count as graded). Anything else - empty,
/// `Requires grading`, `N/A` - is left ungraded (`None`).
fn prefilled_grade(cell: &str) -> Option<String> {
    let t = cell.trim();
    if t == "-" {
        return Some("0".to_string());
    }
    crate::grade::evaluate_grade(t).map(|_| t.to_string())
}

/// Row ids that already have a grade for `(exam, col)`.
pub async fn graded_row_ids(
    conn: &Connection,
    exam: &str,
    col: &str,
) -> AppResult<std::collections::HashSet<String>> {
    let mut out = std::collections::HashSet::new();
    let mut r = conn
        .query(
            "SELECT row_id FROM graded_items WHERE exam = ? AND output_col = ?",
            (exam, col),
        )
        .await?;
    while let Some(row) = r.next().await? {
        out.insert(row.get::<String>(0)?);
    }
    Ok(out)
}

/// Seed grades already implied by the sheet, for the given output columns. Per
/// row, in priority order: a `prefilled` value in the output cell (numeric, or
/// `-` => 0); else, for a blank/`-`/`N/A` *response* with no grade yet, an
/// `auto-zero` `0` so unanswered students still export with a grade (matching the
/// legacy auto-zero sweep). Rows already graded are never clobbered. Run on exam
/// create, column-expand and before results/export; idempotent. Returns the
/// number of grades seeded.
pub async fn seed_prefilled_grades(
    conn: &Connection,
    exam: &Exam,
    rows: &[HashMap<String, String>],
    cols: &[String],
) -> AppResult<usize> {
    let mut seeded = 0usize;
    for col in cols {
        let Some(q) = exam.questions.get(col) else { continue };
        let input_col = q.input_column.clone();
        let already = graded_row_ids(conn, &exam.name, col).await?;
        for row in rows {
            let row_id = crate::domain::row_id(row, &exam.id_columns);
            let input_text = row.get(&input_col).cloned().unwrap_or_default();
            let (grade, source) =
                if let Some(g) = prefilled_grade(&row.get(col).cloned().unwrap_or_default()) {
                    (g, "prefilled")
                } else if !crate::domain::is_meaningful(&input_text) && !already.contains(&row_id) {
                    ("0".to_string(), "auto-zero")
                } else {
                    continue;
                };
            let item = GradedItem {
                row_id,
                input_text,
                grade,
                timestamp: now_iso(),
            };
            put_graded_item(conn, &exam.name, col, &item, source, DEFAULT_SESSION).await?;
            seeded += 1;
        }
    }
    Ok(seeded)
}

/// Current (grade, source) for a cell, if graded.
pub async fn get_graded(
    conn: &Connection,
    exam: &str,
    col: &str,
    row_id: &str,
) -> AppResult<Option<(String, String)>> {
    let mut r = conn
        .query(
            "SELECT grade, source FROM graded_items \
             WHERE exam = ? AND output_col = ? AND row_id = ?",
            (exam, col, row_id),
        )
        .await?;
    if let Some(row) = r.next().await? {
        Ok(Some((row.get(0)?, row.get(1)?)))
    } else {
        Ok(None)
    }
}

pub async fn add_conflict(conn: &Connection, exam: &str, c: &GradeConflict) -> AppResult<()> {
    conn.execute(
        "DELETE FROM grade_conflicts WHERE exam = ? AND output_col = ? AND row_id = ? AND incoming_source = ?",
        (exam, c.output_col.as_str(), c.row_id.as_str(), c.incoming_source.as_str()),
    )
    .await?;
    conn.execute(
        "INSERT INTO grade_conflicts (exam, output_col, row_id, existing_grade, existing_source, incoming_grade, incoming_source, input_text, timestamp) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            exam,
            c.output_col.as_str(),
            c.row_id.as_str(),
            c.existing_grade.as_str(),
            c.existing_source.as_str(),
            c.incoming_grade.as_str(),
            c.incoming_source.as_str(),
            c.input_text.as_str(),
            c.timestamp.as_str(),
        ),
    )
    .await?;
    Ok(())
}

pub async fn list_conflicts(conn: &Connection, exam: &str) -> AppResult<Vec<GradeConflict>> {
    let mut out = Vec::new();
    let mut r = conn
        .query(
            "SELECT output_col, row_id, existing_grade, existing_source, incoming_grade, incoming_source, input_text, timestamp \
             FROM grade_conflicts WHERE exam = ? ORDER BY output_col, row_id",
            (exam,),
        )
        .await?;
    while let Some(row) = r.next().await? {
        out.push(GradeConflict {
            output_col: row.get(0)?,
            row_id: row.get(1)?,
            existing_grade: row.get(2)?,
            existing_source: row.get(3)?,
            incoming_grade: row.get(4)?,
            incoming_source: row.get(5)?,
            input_text: row.get(6)?,
            timestamp: row.get(7)?,
        });
    }
    Ok(out)
}

pub async fn count_conflicts(conn: &Connection, exam: &str) -> AppResult<usize> {
    let mut c = conn
        .query("SELECT COUNT(*) FROM grade_conflicts WHERE exam = ?", (exam,))
        .await?;
    Ok(c.next().await?.map(|r| r.get::<i64>(0)).transpose()?.unwrap_or(0) as usize)
}

pub async fn clear_conflicts_for(
    conn: &Connection,
    exam: &str,
    col: &str,
    row_id: &str,
) -> AppResult<()> {
    conn.execute(
        "DELETE FROM grade_conflicts WHERE exam = ? AND output_col = ? AND row_id = ?",
        (exam, col, row_id),
    )
    .await?;
    Ok(())
}

pub async fn delete_graded_item(
    conn: &Connection,
    exam: &str,
    col: &str,
    row_id: &str,
) -> AppResult<()> {
    conn.execute(
        "DELETE FROM graded_items WHERE exam = ? AND output_col = ? AND row_id = ?",
        (exam, col, row_id),
    )
    .await?;
    Ok(())
}

async fn question_gq_id(conn: &Connection, exam: &str, col: &str) -> AppResult<Option<String>> {
    let mut r = conn
        .query(
            "SELECT global_question_id FROM questions WHERE exam = ? AND output_col = ?",
            (exam, col),
        )
        .await?;
    if let Some(row) = r.next().await? {
        Ok(opt(row.get::<String>(0)?))
    } else {
        Ok(None)
    }
}

pub async fn set_archived(conn: &Connection, exam: &str, archived: bool) -> AppResult<bool> {
    let existed = exam_exists(conn, exam).await?;
    if existed {
        conn.execute(
            "UPDATE exams SET archived = ? WHERE name = ?",
            (i64::from(archived), exam),
        )
        .await?;
    }
    Ok(existed)
}

pub async fn set_course(conn: &Connection, exam: &str, course: Option<&str>) -> AppResult<()> {
    conn.execute(
        "UPDATE exams SET course = ? WHERE name = ?",
        (course.unwrap_or(""), exam),
    )
    .await?;
    Ok(())
}

/// Replace an exam's column mapping (id/input/output). Existing questions and
/// their grades are left intact; callers `ensure_question` for any new output
/// columns and upsert them separately.
pub async fn set_columns(
    conn: &Connection,
    exam: &str,
    id_columns: &[String],
    input_columns: &[String],
    output_columns: &[String],
) -> AppResult<()> {
    conn.execute(
        "UPDATE exams SET id_columns = ?, input_columns = ?, output_columns = ? WHERE name = ?",
        (
            json_array(id_columns),
            json_array(input_columns),
            json_array(output_columns),
            exam,
        ),
    )
    .await?;
    Ok(())
}

/// Bump an exam's last-updated timestamp.
pub async fn touch(conn: &Connection, exam: &str) -> AppResult<()> {
    conn.execute(
        "UPDATE exams SET last_updated = ? WHERE name = ?",
        (now_iso(), exam),
    )
    .await?;
    Ok(())
}

pub async fn delete_exam(conn: &Connection, name: &str) -> AppResult<bool> {
    let existed = exam_exists(conn, name).await?;
    for sql in [
        "DELETE FROM graded_items WHERE exam = ?",
        "DELETE FROM caches WHERE exam = ?",
        "DELETE FROM grade_conflicts WHERE exam = ?",
        "DELETE FROM questions WHERE exam = ?",
        "DELETE FROM sessions WHERE exam = ?",
        "DELETE FROM exams WHERE name = ?",
    ] {
        conn.execute(sql, (name,)).await?;
    }
    Ok(existed)
}

async fn load_graded_items(conn: &Connection, exam: &str, col: &str) -> AppResult<Vec<GradedItem>> {
    let mut items = Vec::new();
    let mut g = conn
        .query(
            "SELECT row_id, input_text, grade, timestamp FROM graded_items \
             WHERE exam = ? AND output_col = ? ORDER BY timestamp, row_id",
            (exam, col),
        )
        .await?;
    while let Some(row) = g.next().await? {
        items.push(GradedItem {
            row_id: row.get(0)?,
            input_text: row.get(1)?,
            grade: row.get(2)?,
            timestamp: row.get(3)?,
        });
    }
    Ok(items)
}

fn question_from_row(row: &turso::Row) -> AppResult<(String, QuestionGrades)> {
    let col: String = row.get(0)?;
    let gq: String = row.get(5)?;
    let sr: String = row.get(6)?;
    let estimate: String = row.get(12)?;
    let q = QuestionGrades {
        question_name: row.get(1)?,
        input_column: row.get(2)?,
        exam_question: row.get(3)?,
        sample_answer: row.get(4)?,
        global_question_id: opt(gq),
        graded_items: Vec::new(),
        sampling_result: if sr.is_empty() {
            None
        } else {
            serde_json::from_str(&sr).ok()
        },
        var: row.get(7)?,
        group: row.get(8)?,
        qtype: row.get(9)?,
        max_points: row.get(10)?,
        position: row.get::<i64>(11)? as i32,
        estimate: opt(estimate),
        external_graded_items: Vec::new(),
    };
    Ok((col, q))
}

const Q_COLS: &str = "output_col, question_name, input_column, exam_question, \
     sample_answer, global_question_id, sampling_result, var, qgroup, qtype, \
     max_points, position, estimate";

pub async fn load_exam(conn: &Connection, name: &str) -> AppResult<Option<Exam>> {
    let meta = {
        let mut r = conn
            .query(
                "SELECT exam_file, id_columns, input_columns, output_columns, course, last_updated, scheme \
                 FROM exams WHERE name = ?",
                (name,),
            )
            .await?;
        match r.next().await? {
            Some(row) => {
                let exam_file: String = row.get(0)?;
                let id_columns = parse_array(&row.get::<String>(1)?);
                let input_columns = parse_array(&row.get::<String>(2)?);
                let output_columns = parse_array(&row.get::<String>(3)?);
                let course = opt(row.get::<String>(4)?);
                let last_updated: String = row.get(5)?;
                let scheme_s: String = row.get(6)?;
                let scheme = if scheme_s.is_empty() {
                    None
                } else {
                    serde_json::from_str(&scheme_s).ok()
                };
                (exam_file, id_columns, input_columns, output_columns, course, last_updated, scheme)
            }
            None => return Ok(None),
        }
    };

    // Question skeletons first (drop the Rows before per-question queries).
    let mut questions: HashMap<String, QuestionGrades> = HashMap::new();
    {
        let sql = format!("SELECT {Q_COLS} FROM questions WHERE exam = ?");
        let mut qr = conn.query(&sql, (name,)).await?;
        while let Some(row) = qr.next().await? {
            let (col, q) = question_from_row(&row)?;
            questions.insert(col, q);
        }
    }

    for (col, q) in questions.iter_mut() {
        q.graded_items = load_graded_items(conn, name, col).await?;
        hydrate_external(conn, q, Some(name)).await?;
    }

    Ok(Some(Exam {
        name: name.to_string(),
        exam_file: meta.0,
        id_columns: meta.1,
        input_columns: meta.2,
        output_columns: meta.3,
        course: meta.4,
        last_updated: meta.5,
        questions,
        scheme: meta.6,
    }))
}

/// Persist (or clear) the examination grade scheme.
pub async fn set_scheme(
    conn: &Connection,
    exam: &str,
    scheme: &Option<crate::scheme::GradeScheme>,
) -> AppResult<()> {
    let s = scheme
        .as_ref()
        .map(|x| serde_json::to_string(x).unwrap_or_default())
        .unwrap_or_default();
    conn.execute("UPDATE exams SET scheme = ? WHERE name = ?", (s, exam)).await?;
    Ok(())
}

pub async fn load_question(
    conn: &Connection,
    exam: &str,
    col: &str,
) -> AppResult<Option<QuestionGrades>> {
    let mut q = {
        let sql = format!("SELECT {Q_COLS} FROM questions WHERE exam = ? AND output_col = ?");
        let mut r = conn.query(&sql, (exam, col)).await?;
        match r.next().await? {
            Some(row) => question_from_row(&row)?.1,
            None => return Ok(None),
        }
    };
    q.graded_items = load_graded_items(conn, exam, col).await?;
    hydrate_external(conn, &mut q, Some(exam)).await?;
    Ok(Some(q))
}

pub async fn list_exams(conn: &Connection, archived: bool) -> AppResult<Vec<ExamSummary>> {
    struct Row {
        name: String,
        file: String,
        course: String,
        updated: String,
    }
    let mut tmp = Vec::new();
    {
        let mut r = conn
            .query(
                "SELECT name, exam_file, course, last_updated FROM exams \
                 WHERE archived = ? ORDER BY last_updated DESC",
                (i64::from(archived),),
            )
            .await?;
        while let Some(row) = r.next().await? {
            tmp.push(Row {
                name: row.get(0)?,
                file: row.get(1)?,
                course: row.get(2)?,
                updated: row.get(3)?,
            });
        }
    }
    let mut out = Vec::with_capacity(tmp.len());
    for t in tmp {
        let num_questions = {
            let mut c = conn
                .query(
                    "SELECT COUNT(*) FROM questions WHERE exam = ?",
                    (t.name.as_str(),),
                )
                .await?;
            c.next().await?.map(|r| r.get::<i64>(0)).transpose()?.unwrap_or(0) as usize
        };
        let graded_count = {
            let mut c = conn
                .query(
                    "SELECT COUNT(*) FROM graded_items WHERE exam = ?",
                    (t.name.as_str(),),
                )
                .await?;
            c.next().await?.map(|r| r.get::<i64>(0)).transpose()?.unwrap_or(0) as usize
        };
        out.push(ExamSummary {
            name: t.name,
            exam_file: t.file,
            course: opt(t.course),
            last_updated: t.updated,
            num_questions,
            graded_count,
            archived,
        });
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Sessions (lightweight grading passes under an exam)
// ---------------------------------------------------------------------------

pub async fn session_exists(conn: &Connection, exam: &str, name: &str) -> AppResult<bool> {
    let mut r = conn
        .query("SELECT 1 FROM sessions WHERE exam = ? AND name = ?", (exam, name))
        .await?;
    Ok(r.next().await?.is_some())
}

/// Create a session if it doesn't already exist; returns it either way.
pub async fn create_session(conn: &Connection, exam: &str, name: &str) -> AppResult<Session> {
    let now = now_iso();
    if !session_exists(conn, exam, name).await? {
        conn.execute(
            "INSERT INTO sessions (exam, name, created_at, last_updated) VALUES (?, ?, ?, ?)",
            (exam, name, now.as_str(), now.as_str()),
        )
        .await?;
    }
    let mut r = conn
        .query(
            "SELECT created_at, last_updated FROM sessions WHERE exam = ? AND name = ?",
            (exam, name),
        )
        .await?;
    let (created_at, last_updated) = match r.next().await? {
        Some(row) => (row.get::<String>(0)?, row.get::<String>(1)?),
        None => (now.clone(), now),
    };
    Ok(Session {
        exam: exam.to_string(),
        name: name.to_string(),
        created_at,
        last_updated,
    })
}

pub async fn touch_session(conn: &Connection, exam: &str, name: &str) -> AppResult<()> {
    conn.execute(
        "UPDATE sessions SET last_updated = ? WHERE exam = ? AND name = ?",
        (now_iso(), exam, name),
    )
    .await?;
    Ok(())
}

pub async fn delete_session(conn: &Connection, exam: &str, name: &str) -> AppResult<bool> {
    let existed = session_exists(conn, exam, name).await?;
    conn.execute(
        "DELETE FROM sessions WHERE exam = ? AND name = ?",
        (exam, name),
    )
    .await?;
    Ok(existed)
}

pub async fn list_sessions(conn: &Connection, exam: &str) -> AppResult<Vec<SessionSummary>> {
    struct Row {
        name: String,
        created: String,
        updated: String,
    }
    let mut tmp = Vec::new();
    {
        let mut r = conn
            .query(
                "SELECT name, created_at, last_updated FROM sessions \
                 WHERE exam = ? ORDER BY created_at",
                (exam,),
            )
            .await?;
        while let Some(row) = r.next().await? {
            tmp.push(Row {
                name: row.get(0)?,
                created: row.get(1)?,
                updated: row.get(2)?,
            });
        }
    }
    let mut out = Vec::with_capacity(tmp.len());
    for t in tmp {
        let graded_count = {
            let mut c = conn
                .query(
                    "SELECT COUNT(*) FROM graded_items WHERE exam = ? AND session = ?",
                    (exam, t.name.as_str()),
                )
                .await?;
            c.next().await?.map(|r| r.get::<i64>(0)).transpose()?.unwrap_or(0) as usize
        };
        out.push(SessionSummary {
            exam: exam.to_string(),
            name: t.name,
            created_at: t.created,
            last_updated: t.updated,
            graded_count,
        });
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Feature cache (Turso)
// ---------------------------------------------------------------------------

pub async fn has_feature_vector(
    conn: &Connection,
    exam: &str,
    input_column: &str,
    row_id: &str,
) -> AppResult<bool> {
    let mut r = conn
        .query(
            "SELECT 1 FROM caches WHERE exam = ? AND kind = 'features' AND input_column = ? AND row_id = ?",
            (exam, input_column, row_id),
        )
        .await?;
    Ok(r.next().await?.is_some())
}

pub async fn put_feature_vector(
    conn: &Connection,
    exam: &str,
    input_column: &str,
    row_id: &str,
    vector: &[f32],
) -> AppResult<()> {
    conn.execute(
        "DELETE FROM caches WHERE exam = ? AND kind = 'features' AND input_column = ? AND row_id = ?",
        (exam, input_column, row_id),
    )
    .await?;
    conn.execute(
        "INSERT INTO caches (exam, kind, input_column, row_id, vector) \
         VALUES (?, 'features', ?, ?, ?)",
        (exam, input_column, row_id, db::f32s_to_blob(vector)),
    )
    .await?;
    Ok(())
}

pub async fn load_feature_cache(
    conn: &Connection,
    exam: &str,
    input_column: &str,
) -> AppResult<HashMap<String, Vec<f32>>> {
    let mut out = HashMap::new();
    let mut r = conn
        .query(
            "SELECT row_id, vector FROM caches WHERE exam = ? AND kind = 'features' AND input_column = ?",
            (exam, input_column),
        )
        .await?;
    while let Some(row) = r.next().await? {
        let rid: String = row.get(0)?;
        let blob: Vec<u8> = row.get(1)?;
        out.insert(rid, db::blob_to_f32s(&blob));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Global question bank embeddings (Turso) - app-wide, not exam-scoped
// ---------------------------------------------------------------------------

pub async fn put_bank_vector(
    conn: &Connection,
    bank: &str,
    qid: &str,
    lang: &str,
    vector: &[f32],
) -> AppResult<()> {
    conn.execute(
        "DELETE FROM global_bank_vectors WHERE bank = ? AND qid = ? AND lang = ?",
        (bank, qid, lang),
    )
    .await?;
    conn.execute(
        "INSERT INTO global_bank_vectors (bank, qid, lang, vector) VALUES (?, ?, ?, ?)",
        (bank, qid, lang, db::f32s_to_blob(vector)),
    )
    .await?;
    Ok(())
}

/// All cached bank vectors for `lang`, keyed by `(bank, qid)`.
pub async fn load_bank_vectors(
    conn: &Connection,
    lang: &str,
) -> AppResult<HashMap<(String, String), Vec<f32>>> {
    let mut out = HashMap::new();
    let mut r = conn
        .query(
            "SELECT bank, qid, vector FROM global_bank_vectors WHERE lang = ?",
            (lang,),
        )
        .await?;
    while let Some(row) = r.next().await? {
        let bank: String = row.get(0)?;
        let qid: String = row.get(1)?;
        let blob: Vec<u8> = row.get(2)?;
        out.insert((bank, qid), db::blob_to_f32s(&blob));
    }
    Ok(out)
}

/// Number of cached bank vectors (across all languages).
pub async fn count_bank_vectors(conn: &Connection) -> AppResult<usize> {
    let mut r = conn.query("SELECT COUNT(*) FROM global_bank_vectors", ()).await?;
    Ok(r.next().await?.map(|row| row.get::<i64>(0)).transpose()?.unwrap_or(0) as usize)
}

/// Remove every question (and its vectors) for a bank - used to fully replace a
/// bank on re-import.
pub async fn delete_bank(conn: &Connection, bank: &str) -> AppResult<()> {
    conn.execute("DELETE FROM global_bank_questions WHERE bank = ?", (bank,)).await?;
    conn.execute("DELETE FROM global_bank_vectors WHERE bank = ?", (bank,)).await?;
    Ok(())
}

pub async fn upsert_bank_question(conn: &Connection, q: &BankQuestion) -> AppResult<()> {
    conn.execute(
        "DELETE FROM global_bank_questions WHERE bank = ? AND qid = ?",
        (q.bank.as_str(), q.qid.as_str()),
    )
    .await?;
    conn.execute(
        "INSERT INTO global_bank_questions \
         (bank, qid, q_se, q_en, ans_se, ans_en, chapter, subject, qtype) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            q.bank.as_str(),
            q.qid.as_str(),
            q.q_se.as_str(),
            q.q_en.as_str(),
            q.ans_se.as_str(),
            q.ans_en.as_str(),
            q.chapter.as_str(),
            q.subject.as_str(),
            q.qtype.as_str(),
        ),
    )
    .await?;
    Ok(())
}

/// All bank questions, ordered by bank then qid.
pub async fn load_bank_questions(conn: &Connection) -> AppResult<Vec<BankQuestion>> {
    let mut out = Vec::new();
    let mut r = conn
        .query(
            "SELECT bank, qid, q_se, q_en, ans_se, ans_en, chapter, subject, qtype \
             FROM global_bank_questions ORDER BY bank, qid",
            (),
        )
        .await?;
    while let Some(row) = r.next().await? {
        out.push(BankQuestion {
            bank: row.get(0)?,
            qid: row.get(1)?,
            q_se: row.get(2)?,
            q_en: row.get(3)?,
            ans_se: row.get(4)?,
            ans_en: row.get(5)?,
            chapter: row.get(6)?,
            subject: row.get(7)?,
            qtype: row.get(8)?,
        });
    }
    Ok(out)
}

/// Per-bank question counts, ordered by bank.
pub async fn bank_counts(conn: &Connection) -> AppResult<Vec<(String, usize)>> {
    let mut out = Vec::new();
    let mut r = conn
        .query(
            "SELECT bank, COUNT(*) FROM global_bank_questions GROUP BY bank ORDER BY bank",
            (),
        )
        .await?;
    while let Some(row) = r.next().await? {
        out.push((row.get::<String>(0)?, row.get::<i64>(1)? as usize));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Cross-exam graded pool (Turso)
// ---------------------------------------------------------------------------

pub async fn hydrate_external(
    conn: &Connection,
    question: &mut QuestionGrades,
    current_exam: Option<&str>,
) -> AppResult<()> {
    let Some(gq) = question.global_question_id.clone() else {
        question.external_graded_items = Vec::new();
        return Ok(());
    };
    let current = current_exam.unwrap_or("");
    let mut items = Vec::new();
    let mut r = conn
        .query(
            "SELECT source_exam, row_id, input_text, grade, timestamp FROM graded_pool \
             WHERE global_question_id = ? AND source_exam <> ?",
            (gq.as_str(), current),
        )
        .await?;
    while let Some(row) = r.next().await? {
        let src: String = row.get(0)?;
        let rid: String = row.get(1)?;
        items.push(GradedItem {
            row_id: format!("{src}::{rid}"),
            input_text: row.get(2)?,
            grade: row.get(3)?,
            timestamp: row.get(4)?,
        });
    }
    question.external_graded_items = items;
    Ok(())
}

/// Push a linked question's existing grades into the cross-exam pool. Call after
/// setting its `global_question_id` so prior grades (recorded before the link)
/// become ICL examples for other exams sharing that id. No-op if unlinked.
pub async fn sync_question_grades_to_pool(
    conn: &Connection,
    exam: &str,
    col: &str,
) -> AppResult<()> {
    let Some(gq) = question_gq_id(conn, exam, col).await? else {
        return Ok(());
    };
    for item in load_graded_items(conn, exam, col).await? {
        sync_item_to_pool(conn, &gq, exam, &item).await?;
    }
    Ok(())
}

async fn sync_item_to_pool(
    conn: &Connection,
    gq_id: &str,
    source_exam: &str,
    item: &GradedItem,
) -> AppResult<()> {
    let exists = {
        let mut r = conn
            .query(
                "SELECT 1 FROM graded_pool WHERE global_question_id = ? AND source_exam = ? AND row_id = ?",
                (gq_id, source_exam, item.row_id.as_str()),
            )
            .await?;
        r.next().await?.is_some()
    };
    if exists {
        return Ok(());
    }
    conn.execute(
        "INSERT INTO graded_pool (global_question_id, source_exam, row_id, input_text, grade, timestamp) \
         VALUES (?, ?, ?, ?, ?, ?)",
        (
            gq_id,
            source_exam,
            item.row_id.as_str(),
            item.input_text.as_str(),
            item.grade.as_str(),
            item.timestamp.as_str(),
        ),
    )
    .await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Legacy import (old Python-app JSON -> new exam-centric DB, on demand)
// ---------------------------------------------------------------------------

/// The pre-Rust on-disk session shape. Field names differ from `Exam`
/// (`session_name`/`csv_file`), so it is read into this struct then mapped.
/// `QuestionGrades` is unchanged, so old `questions` deserialize directly;
/// unknown legacy fields (e.g. a per-item `embedding`) are ignored.
#[derive(serde::Deserialize, Default)]
struct LegacySession {
    #[serde(default)]
    session_name: String,
    #[serde(default)]
    csv_file: String,
    #[serde(default)]
    id_columns: Vec<String>,
    #[serde(default)]
    input_columns: Vec<String>,
    #[serde(default)]
    output_columns: Vec<String>,
    #[serde(default)]
    course: Option<String>,
    #[serde(default)]
    last_updated: String,
    #[serde(default)]
    questions: HashMap<String, QuestionGrades>,
    #[serde(default)]
    scheme: Option<crate::scheme::GradeScheme>,
}

async fn unique_exam_name(conn: &Connection, orig: &str, suffix: &str) -> AppResult<String> {
    let base = format!("{orig}_{suffix}");
    if !exam_exists(conn, &base).await? {
        return Ok(base);
    }
    let mut i = 2;
    loop {
        let candidate = format!("{base}_{i}");
        if !exam_exists(conn, &candidate).await? {
            return Ok(candidate);
        }
        i += 1;
    }
}

/// Import one legacy session JSON value as an exam (+ a `default` session) in the
/// new format. Returns the stored exam name. On name collision the exam is
/// suffixed with `suffix` rather than overwriting. `caches_dir`/`stem` locate the
/// side-car `<stem>.cache.json`.
pub async fn import_legacy_session(
    conn: &Connection,
    value: &Value,
    stem: &str,
    caches_dir: &Path,
    course_override: Option<&str>,
    suffix: &str,
) -> AppResult<String> {
    let legacy: LegacySession = serde_json::from_value(value.clone()).unwrap_or_default();
    let orig = if legacy.session_name.is_empty() {
        stem.to_string()
    } else {
        legacy.session_name.clone()
    };
    let name = if exam_exists(conn, &orig).await? {
        unique_exam_name(conn, &orig, suffix).await?
    } else {
        orig
    };

    let exam = Exam {
        name: name.clone(),
        exam_file: legacy.csv_file,
        id_columns: legacy.id_columns,
        input_columns: legacy.input_columns,
        output_columns: legacy.output_columns,
        course: course_override.map(str::to_string).or(legacy.course),
        last_updated: if legacy.last_updated.is_empty() {
            now_iso()
        } else {
            legacy.last_updated
        },
        questions: legacy.questions,
        scheme: legacy.scheme,
    };
    insert_exam(conn, &exam).await?;
    create_session(conn, &name, DEFAULT_SESSION).await?;
    import_caches(conn, &name, stem, caches_dir, value).await?;
    Ok(name)
}

/// Import a legacy session's feature-vector cache (side-car `<stem>.cache.json`
/// preferred, else embedded `features_cache`) into the new `caches` table.
async fn import_caches(
    conn: &Connection,
    exam: &str,
    stem: &str,
    caches_dir: &Path,
    value: &Value,
) -> AppResult<()> {
    let side = caches_dir.join(format!("{stem}.cache.json"));
    let features: Option<Value> = if side.exists() {
        std::fs::read_to_string(&side)
            .ok()
            .and_then(|s| serde_json::from_str::<Value>(&s).ok())
            .and_then(|v| v.get("features_cache").cloned())
    } else {
        value.get("features_cache").cloned()
    };
    let Some(obj) = features.as_ref().and_then(|v| v.as_object()) else {
        return Ok(());
    };
    for (input_col, rowmap) in obj {
        let Some(rm) = rowmap.as_object() else { continue };
        for (rid, arr) in rm {
            let Some(a) = arr.as_array() else { continue };
            let v: Vec<f32> = a.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect();
            if !v.is_empty() {
                put_feature_vector(conn, exam, input_col, rid, &v).await?;
            }
        }
    }
    Ok(())
}

/// Count importable legacy session JSON files in a directory.
pub fn count_session_files(dir: &Path) -> usize {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries
        .flatten()
        .filter(|e| {
            let n = e.file_name().to_string_lossy().to_string();
            n.ends_with(".json") && !n.ends_with(".cache.json")
        })
        .count()
}

/// Import every legacy session `*.json` in a directory as an exam in the new
/// format. `archived` marks the created exams as archived (for `archive/`).
/// Returns the stored exam names.
pub async fn import_session_dir(
    conn: &Connection,
    dir: &Path,
    archived: bool,
) -> AppResult<Vec<String>> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Ok(out);
    };
    for e in entries.flatten() {
        let fname = e.file_name().to_string_lossy().to_string();
        if !fname.ends_with(".json") || fname.ends_with(".cache.json") {
            continue;
        }
        let stem = fname.trim_end_matches(".json");
        let Ok(raw) = std::fs::read_to_string(e.path()) else {
            continue;
        };
        let Ok(value) = serde_json::from_str::<Value>(&raw) else {
            continue;
        };
        let name = import_legacy_session(conn, &value, stem, dir, None, "import").await?;
        if archived {
            set_archived(conn, &name, true).await?;
        }
        out.push(name);
    }
    Ok(out)
}

/// Merge a legacy `graded_pool/<gq_id>.jsonl` directory into the cross-exam pool.
pub async fn import_pool_dir(conn: &Connection, dir: &Path) -> AppResult<()> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Ok(());
    };
    for e in entries.flatten() {
        let fname = e.file_name().to_string_lossy().to_string();
        if !fname.ends_with(".jsonl") {
            continue;
        }
        let gq = fname.trim_end_matches(".jsonl").to_string();
        let Ok(content) = std::fs::read_to_string(e.path()) else {
            continue;
        };
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let Ok(rec) = serde_json::from_str::<Value>(line) else {
                continue;
            };
            // Legacy jsonl tags the source with `source_session`.
            let src = rec
                .get("source_session")
                .or_else(|| rec.get("source_exam"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if src.is_empty() {
                continue;
            }
            let item = GradedItem {
                row_id: rec.get("row_id").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                input_text: rec.get("input_text").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                grade: rec.get("grade").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                timestamp: rec.get("timestamp").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            };
            sync_item_to_pool(conn, &gq, src, &item).await?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Excel export (on disk)
// ---------------------------------------------------------------------------

/// Write a simple sheet (header row + string data rows) to an xlsx file on disk.
/// Used to materialise derived exam files (e.g. the combined Moodle dump).
pub fn write_xlsx(path: &Path, headers: &[String], rows: &[Vec<String>]) -> AppResult<()> {
    use rust_xlsxwriter::Workbook;
    let mut workbook = Workbook::new();
    let sheet = workbook.add_worksheet();
    for (c, h) in headers.iter().enumerate() {
        sheet
            .write_string(0, c as u16, h)
            .map_err(|e| AppError::Other(anyhow::anyhow!(e)))?;
    }
    for (r, row) in rows.iter().enumerate() {
        for (c, val) in row.iter().enumerate() {
            sheet
                .write_string((r + 1) as u32, c as u16, val)
                .map_err(|e| AppError::Other(anyhow::anyhow!(e)))?;
        }
    }
    let bytes = workbook
        .save_to_buffer()
        .map_err(|e| AppError::Other(anyhow::anyhow!(e)))?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, bytes)?;
    Ok(())
}

/// Stem of an exam's file, for naming downloads.
fn exam_file_stem(exam: &Exam) -> String {
    Path::new(&exam.exam_file)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("exam")
        .to_string()
}

/// Build the full graded xlsx in memory. Returns (download filename, bytes).
pub fn build_graded_xlsx(
    exam: &Exam,
    exam_rows: &[HashMap<String, String>],
    columns: &[String],
) -> AppResult<(String, Vec<u8>)> {
    use rust_xlsxwriter::Workbook;

    let filename = format!("{}.xlsx", exam_file_stem(exam));

    let mut grades_by_row: HashMap<String, HashMap<String, String>> = HashMap::new();
    for (col, q) in &exam.questions {
        for item in &q.graded_items {
            let cell = match crate::grade::evaluate_grade(&item.grade) {
                Some(total) => format_total(total),
                None => item.grade.clone(),
            };
            grades_by_row
                .entry(item.row_id.clone())
                .or_default()
                .insert(col.clone(), cell);
        }
    }

    let mut workbook = Workbook::new();
    let sheet = workbook.add_worksheet();
    for (c, header) in columns.iter().enumerate() {
        sheet
            .write_string(0, c as u16, header)
            .map_err(|e| AppError::Other(anyhow::anyhow!(e)))?;
    }
    for (r, row) in exam_rows.iter().enumerate() {
        let rid = crate::domain::row_id(row, &exam.id_columns);
        let overrides = grades_by_row.get(&rid);
        for (c, col) in columns.iter().enumerate() {
            let value = overrides
                .and_then(|o| o.get(col))
                .cloned()
                .or_else(|| row.get(col).cloned())
                .unwrap_or_default();
            sheet
                .write_string((r + 1) as u32, c as u16, &value)
                .map_err(|e| AppError::Other(anyhow::anyhow!(e)))?;
        }
    }
    let bytes = workbook
        .save_to_buffer()
        .map_err(|e| AppError::Other(anyhow::anyhow!(e)))?;
    Ok((filename, bytes))
}

fn format_total(total: f64) -> String {
    if total.fract() == 0.0 && total.is_finite() {
        format!("{}", total as i64)
    } else {
        format!("{total}")
    }
}

/// Daisy grade-import sheet: no header, column A = id, column B = final grade,
/// one row per student sorted by id (matches SU's Daisy import format).
pub fn build_daisy_xlsx(
    exam: &Exam,
    results: &[crate::scheme::StudentResult],
) -> AppResult<(String, Vec<u8>)> {
    use rust_xlsxwriter::Workbook;
    let filename = format!("{}_daisy_import.xlsx", exam_file_stem(exam));

    let mut sorted: Vec<&crate::scheme::StudentResult> = results.iter().collect();
    sorted.sort_by(|a, b| a.id.cmp(&b.id));

    let mut workbook = Workbook::new();
    let sheet = workbook.add_worksheet();
    for (r, sr) in sorted.iter().enumerate() {
        sheet
            .write_string(r as u32, 0, &sr.id)
            .map_err(|e| AppError::Other(anyhow::anyhow!(e)))?;
        sheet
            .write_string(r as u32, 1, &sr.grade)
            .map_err(|e| AppError::Other(anyhow::anyhow!(e)))?;
    }
    let bytes = workbook
        .save_to_buffer()
        .map_err(|e| AppError::Other(anyhow::anyhow!(e)))?;
    Ok((filename, bytes))
}

/// Per-question CSV: id columns, each question's points, total and final grade.
pub fn build_per_question_csv(
    exam: &Exam,
    exam_rows: &[HashMap<String, String>],
    results: &[crate::scheme::StudentResult],
) -> AppResult<(String, Vec<u8>)> {
    let filename = format!("{}_per_question.csv", exam_file_stem(exam));

    let by_id: HashMap<&str, &crate::scheme::StudentResult> =
        results.iter().map(|r| (r.id.as_str(), r)).collect();

    let mut w = csv::Writer::from_writer(Vec::new());
    let mut header: Vec<String> = exam.id_columns.clone();
    header.extend(exam.output_columns.iter().cloned());
    header.push("total".to_string());
    header.push("grade".to_string());
    w.write_record(&header)?;

    for row in exam_rows {
        let rid = crate::domain::row_id(row, &exam.id_columns);
        let mut rec: Vec<String> = exam
            .id_columns
            .iter()
            .map(|c| row.get(c).cloned().unwrap_or_default())
            .collect();
        for col in &exam.output_columns {
            let cell = exam
                .questions
                .get(col)
                .and_then(|q| q.graded_items.iter().find(|gi| gi.row_id == rid))
                .map(|gi| match crate::grade::evaluate_grade(&gi.grade) {
                    Some(t) => format_total(t),
                    None => gi.grade.clone(),
                })
                .unwrap_or_default();
            rec.push(cell);
        }
        let sr = by_id.get(rid.as_str());
        rec.push(sr.map(|r| format_total(r.total)).unwrap_or_default());
        rec.push(sr.map(|r| r.grade.clone()).unwrap_or_default());
        w.write_record(&rec)?;
    }
    w.flush()?;
    let bytes = w
        .into_inner()
        .map_err(|e| AppError::Other(anyhow::anyhow!(e.to_string())))?;
    Ok((filename, bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static CTR: AtomicUsize = AtomicUsize::new(0);

    async fn mem_conn() -> Connection {
        let n = CTR.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("tt-store-db-{}-{}", std::process::id(), n));
        let _ = std::fs::create_dir_all(&dir);
        let database = db::open(dir.join("t.db").to_str().unwrap()).await.unwrap();
        let conn = database.connect().unwrap();
        db::init_schema(&conn).await.unwrap();
        // Leak the dir handle; tests are short-lived and use temp space.
        conn
    }

    fn sample_exam(name: &str) -> Exam {
        Exam {
            name: name.into(),
            exam_file: "e.xlsx".into(),
            id_columns: vec!["id".into()],
            input_columns: vec!["ans".into()],
            output_columns: vec!["g".into()],
            course: Some("CS101".into()),
            last_updated: now_iso(),
            questions: HashMap::new(),
            scheme: None,
        }
    }

    #[tokio::test]
    async fn create_grade_load_roundtrip() {
        let conn = mem_conn().await;
        let mut exam = sample_exam("s1");
        exam.ensure_question("g");
        insert_exam(&conn, &exam).await.unwrap();

        let item = GradedItem {
            row_id: "r1".into(),
            input_text: "hello".into(),
            grade: "2+1.5".into(),
            timestamp: now_iso(),
        };
        put_graded_item(&conn, "s1", "g", &item, "manual", "default").await.unwrap();

        let loaded = load_exam(&conn, "s1").await.unwrap().unwrap();
        assert_eq!(loaded.course.as_deref(), Some("CS101"));
        let q = loaded.questions.get("g").unwrap();
        assert_eq!(q.graded_items.len(), 1);
        assert_eq!(q.graded_items[0].grade, "2+1.5");

        // archive flips listing.
        assert!(list_exams(&conn, false).await.unwrap().iter().any(|s| s.name == "s1"));
        set_archived(&conn, "s1", true).await.unwrap();
        assert!(list_exams(&conn, false).await.unwrap().is_empty());
        assert_eq!(list_exams(&conn, true).await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn sessions_track_graded_counts() {
        let conn = mem_conn().await;
        let mut exam = sample_exam("ex");
        exam.ensure_question("g");
        insert_exam(&conn, &exam).await.unwrap();

        create_session(&conn, "ex", "default").await.unwrap();
        create_session(&conn, "ex", "second").await.unwrap();

        let item = GradedItem {
            row_id: "r1".into(),
            input_text: "x".into(),
            grade: "5".into(),
            timestamp: now_iso(),
        };
        put_graded_item(&conn, "ex", "g", &item, "manual", "second").await.unwrap();

        let sessions = list_sessions(&conn, "ex").await.unwrap();
        assert_eq!(sessions.len(), 2);
        let second = sessions.iter().find(|s| s.name == "second").unwrap();
        assert_eq!(second.graded_count, 1);
        let default = sessions.iter().find(|s| s.name == "default").unwrap();
        assert_eq!(default.graded_count, 0);
    }

    #[tokio::test]
    async fn pool_hydrates_across_exams() {
        let conn = mem_conn().await;
        for name in ["sa", "sb"] {
            let mut e = sample_exam(name);
            e.course = None;
            let q = e.ensure_question("g");
            q.global_question_id = Some("gq7".into());
            insert_exam(&conn, &e).await.unwrap();
        }
        // Grade in sa -> should appear as external for sb.
        put_graded_item(
            &conn,
            "sa",
            "g",
            &GradedItem { row_id: "r1".into(), input_text: "x".into(), grade: "5".into(), timestamp: now_iso() },
            "manual",
            "default",
        )
        .await
        .unwrap();

        let qb = load_question(&conn, "sb", "g").await.unwrap().unwrap();
        assert_eq!(qb.external_graded_items.len(), 1);
        assert_eq!(qb.external_graded_items[0].row_id, "sa::r1");
    }

    #[tokio::test]
    async fn seeds_auto_zero_for_blank_responses() {
        let conn = mem_conn().await;
        let mut exam = sample_exam("az");
        exam.ensure_question("g");
        insert_exam(&conn, &exam).await.unwrap();

        let row = |id: &str, ans: &str| {
            HashMap::from([
                ("id".to_string(), id.to_string()),
                ("ans".to_string(), ans.to_string()),
                ("g".to_string(), String::new()),
            ])
        };
        // r1 answered (must stay ungraded), r2 empty + r3 dash (auto-zeroed).
        let rows = vec![row("r1", "hello"), row("r2", ""), row("r3", "-")];
        // A pre-existing grade on a blank row must not be clobbered.
        put_graded_item(
            &conn,
            "az",
            "g",
            &GradedItem { row_id: "r4".into(), input_text: "".into(), grade: "3".into(), timestamp: now_iso() },
            "imported",
            "default",
        )
        .await
        .unwrap();
        let rows = {
            let mut rows = rows;
            rows.push(row("r4", ""));
            rows
        };

        let seeded = seed_prefilled_grades(&conn, &exam, &rows, &exam.output_columns).await.unwrap();
        assert_eq!(seeded, 2, "only the two unanswered, ungraded rows are zeroed");

        let q = load_question(&conn, "az", "g").await.unwrap().unwrap();
        let grade = |rid: &str| q.graded_items.iter().find(|i| i.row_id == rid).map(|i| i.grade.clone());
        assert_eq!(grade("r1"), None, "answered response is left for grading");
        assert_eq!(grade("r2").as_deref(), Some("0"));
        assert_eq!(grade("r3").as_deref(), Some("0"));
        assert_eq!(grade("r4").as_deref(), Some("3"), "existing grade preserved");
        // Auto-zeroed rows are blank, so they never count toward ICL.
        assert_eq!(q.valid_graded_count(), 0);

        // Idempotent: a second pass seeds nothing new.
        let again = seed_prefilled_grades(&conn, &exam, &rows, &exam.output_columns).await.unwrap();
        assert_eq!(again, 0);
    }

    #[test]
    fn spreadsheet_xml_parses_with_types_and_entities() {
        let xml = r#"<?xml version="1.0"?>
<?mso-application progid="Excel.Sheet"?>
<Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet" xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet">
 <Worksheet ss:Name="Q38 grades">
  <Table>
   <Row><Cell><Data ss:Type="String">daisy_id</Data></Cell><Cell><Data ss:Type="String">pts</Data></Cell></Row>
   <Row><Cell><Data ss:Type="Number">183271</Data></Cell><Cell><Data ss:Type="Number">5.5</Data></Cell></Row>
   <Row><Cell><Data ss:Type="String">A &amp; B</Data></Cell><Cell><Data ss:Type="Number">4.5</Data></Cell></Row>
  </Table>
 </Worksheet>
</Workbook>"#;
        let rows = parse_spreadsheet_xml(xml).unwrap();
        assert_eq!(rows[0], vec!["daisy_id", "pts"]);
        assert_eq!(rows[1], vec!["183271", "5.5"]);
        assert_eq!(rows[2], vec!["A & B", "4.5"]);
    }

    #[tokio::test]
    async fn linking_backfills_pool() {
        let conn = mem_conn().await;
        let mut a = sample_exam("ea");
        a.course = None;
        a.ensure_question("g");
        insert_exam(&conn, &a).await.unwrap();
        // Grade BEFORE the question is linked to a global id.
        put_graded_item(
            &conn,
            "ea",
            "g",
            &GradedItem { row_id: "r1".into(), input_text: "x".into(), grade: "5".into(), timestamp: now_iso() },
            "manual",
            "default",
        )
        .await
        .unwrap();
        // Link it, then backfill the pool.
        let mut q = load_question(&conn, "ea", "g").await.unwrap().unwrap();
        q.global_question_id = Some("gq9".into());
        upsert_question_row(&conn, "ea", "g", &q).await.unwrap();
        sync_question_grades_to_pool(&conn, "ea", "g").await.unwrap();
        // A different exam linked to the same id now sees the prior grade.
        let mut b = sample_exam("eb");
        b.course = None;
        b.ensure_question("g").global_question_id = Some("gq9".into());
        insert_exam(&conn, &b).await.unwrap();
        let loaded = load_question(&conn, "eb", "g").await.unwrap().unwrap();
        assert_eq!(loaded.external_graded_items.len(), 1);
    }

    #[tokio::test]
    async fn feature_cache_blob_roundtrip() {
        let conn = mem_conn().await;
        put_feature_vector(&conn, "s1", "ans", "r1", &[0.5, -1.5, 2.0]).await.unwrap();
        assert!(has_feature_vector(&conn, "s1", "ans", "r1").await.unwrap());
        let cache = load_feature_cache(&conn, "s1", "ans").await.unwrap();
        assert_eq!(cache.get("r1").unwrap(), &vec![0.5, -1.5, 2.0]);
    }
}
