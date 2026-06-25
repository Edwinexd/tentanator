//! Persistence. Session state lives in the Turso database (`db.rs`); exam files
//! and exported Excel stay on disk. Legacy `.tentanator_sessions/*.json`,
//! caches, the single-file session and the graded pool are imported into the DB.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use calamine::{open_workbook_auto, Data, Reader};
use serde_json::Value;
use turso::Connection;

use crate::config::Config;
use crate::db;
use crate::domain::{GradedItem, QuestionGrades, Session, SessionSummary};
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
                if lower.ends_with(".xlsx") || lower.ends_with(".csv") {
                    out.push(name.to_string());
                }
            }
        }
    }
    out.sort();
    out
}

pub fn resolve_exam_path(config: &Config, csv_file: &str) -> Option<PathBuf> {
    let direct = config.exams_dir().join(csv_file);
    if direct.exists() {
        return Some(direct);
    }
    let stem = Path::new(csv_file)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(csv_file);
    for ext in ["xlsx", "csv"] {
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

fn is_xlsx(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()).map(|s| s.to_lowercase()),
        Some(ref e) if e == "xlsx" || e == "xls"
    )
}

pub fn get_exam_columns(path: &Path) -> AppResult<Vec<String>> {
    if is_xlsx(path) {
        let mut wb = open_workbook_auto(path)
            .map_err(|e| AppError::BadRequest(format!("cannot open workbook: {e}")))?;
        let range = wb
            .worksheet_range_at(0)
            .ok_or_else(|| AppError::BadRequest("workbook has no sheets".into()))?
            .map_err(|e| AppError::BadRequest(format!("cannot read sheet: {e}")))?;
        Ok(range
            .rows()
            .next()
            .map(|r| r.iter().map(cell_to_string).collect())
            .unwrap_or_default())
    } else {
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .flexible(true)
            .from_path(path)?;
        Ok(rdr.headers()?.iter().map(|s| s.to_string()).collect())
    }
}

pub fn read_exam_data(path: &Path) -> AppResult<Vec<HashMap<String, String>>> {
    if is_xlsx(path) {
        let mut wb = open_workbook_auto(path)
            .map_err(|e| AppError::BadRequest(format!("cannot open workbook: {e}")))?;
        let range = wb
            .worksheet_range_at(0)
            .ok_or_else(|| AppError::BadRequest("workbook has no sheets".into()))?
            .map_err(|e| AppError::BadRequest(format!("cannot read sheet: {e}")))?;
        let mut rows = range.rows();
        let headers: Vec<String> = rows
            .next()
            .map(|r| r.iter().map(cell_to_string).collect())
            .unwrap_or_default();
        let mut out = Vec::new();
        for r in rows {
            let mut map = HashMap::new();
            for (j, h) in headers.iter().enumerate() {
                map.insert(h.clone(), r.get(j).map(cell_to_string).unwrap_or_default());
            }
            out.push(map);
        }
        Ok(out)
    } else {
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .flexible(true)
            .from_path(path)?;
        let headers: Vec<String> = rdr.headers()?.iter().map(|s| s.to_string()).collect();
        let mut out = Vec::new();
        for rec in rdr.records() {
            let rec = rec?;
            let mut map = HashMap::new();
            for (j, h) in headers.iter().enumerate() {
                map.insert(h.clone(), rec.get(j).unwrap_or("").to_string());
            }
            out.push(map);
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Sessions (Turso)
// ---------------------------------------------------------------------------

pub async fn session_exists(conn: &Connection, name: &str) -> AppResult<bool> {
    let mut r = conn.query("SELECT 1 FROM sessions WHERE name = ?", (name,)).await?;
    Ok(r.next().await?.is_some())
}

pub async fn insert_session(conn: &Connection, session: &Session) -> AppResult<()> {
    conn.execute("BEGIN", ()).await?;
    let res = insert_session_inner(conn, session).await;
    if res.is_ok() {
        conn.execute("COMMIT", ()).await?;
    } else {
        let _ = conn.execute("ROLLBACK", ()).await;
    }
    res
}

async fn insert_session_inner(conn: &Connection, s: &Session) -> AppResult<()> {
    conn.execute("DELETE FROM sessions WHERE name = ?", (s.session_name.as_str(),)).await?;
    conn.execute(
        "INSERT INTO sessions \
         (name, csv_file, id_columns, input_columns, output_columns, course, last_updated, archived) \
         VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
        (
            s.session_name.as_str(),
            s.csv_file.as_str(),
            json_array(&s.id_columns),
            json_array(&s.input_columns),
            json_array(&s.output_columns),
            s.course.clone().unwrap_or_default(),
            s.last_updated.as_str(),
        ),
    )
    .await?;
    for (col, q) in &s.questions {
        upsert_question_row(conn, &s.session_name, col, q).await?;
        for item in &q.graded_items {
            put_graded_item(conn, &s.session_name, col, item).await?;
        }
    }
    Ok(())
}

pub async fn upsert_question_row(
    conn: &Connection,
    name: &str,
    col: &str,
    q: &QuestionGrades,
) -> AppResult<()> {
    conn.execute(
        "DELETE FROM questions WHERE session_name = ? AND output_col = ?",
        (name, col),
    )
    .await?;
    let sr = q
        .sampling_result
        .as_ref()
        .map(|r| serde_json::to_string(r).unwrap_or_default())
        .unwrap_or_default();
    conn.execute(
        "INSERT INTO questions \
         (session_name, output_col, question_name, input_column, exam_question, sample_answer, global_question_id, sampling_result) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            name,
            col,
            q.question_name.as_str(),
            q.input_column.as_str(),
            q.exam_question.as_str(),
            q.sample_answer.as_str(),
            q.global_question_id.clone().unwrap_or_default(),
            sr,
        ),
    )
    .await?;
    Ok(())
}

pub async fn put_graded_item(
    conn: &Connection,
    name: &str,
    col: &str,
    item: &GradedItem,
) -> AppResult<()> {
    conn.execute(
        "DELETE FROM graded_items WHERE session_name = ? AND output_col = ? AND row_id = ?",
        (name, col, item.row_id.as_str()),
    )
    .await?;
    conn.execute(
        "INSERT INTO graded_items (session_name, output_col, row_id, input_text, grade, timestamp) \
         VALUES (?, ?, ?, ?, ?, ?)",
        (
            name,
            col,
            item.row_id.as_str(),
            item.input_text.as_str(),
            item.grade.as_str(),
            item.timestamp.as_str(),
        ),
    )
    .await?;
    // Mirror into the cross-session pool if the question is linked.
    if let Some(gq) = question_gq_id(conn, name, col).await? {
        sync_item_to_pool(conn, &gq, name, item).await?;
    }
    Ok(())
}

pub async fn delete_graded_item(
    conn: &Connection,
    name: &str,
    col: &str,
    row_id: &str,
) -> AppResult<()> {
    conn.execute(
        "DELETE FROM graded_items WHERE session_name = ? AND output_col = ? AND row_id = ?",
        (name, col, row_id),
    )
    .await?;
    Ok(())
}

async fn question_gq_id(conn: &Connection, name: &str, col: &str) -> AppResult<Option<String>> {
    let mut r = conn
        .query(
            "SELECT global_question_id FROM questions WHERE session_name = ? AND output_col = ?",
            (name, col),
        )
        .await?;
    if let Some(row) = r.next().await? {
        Ok(opt(row.get::<String>(0)?))
    } else {
        Ok(None)
    }
}

pub async fn set_archived(conn: &Connection, name: &str, archived: bool) -> AppResult<bool> {
    let existed = session_exists(conn, name).await?;
    if existed {
        conn.execute(
            "UPDATE sessions SET archived = ? WHERE name = ?",
            (i64::from(archived), name),
        )
        .await?;
    }
    Ok(existed)
}

pub async fn set_course(conn: &Connection, name: &str, course: Option<&str>) -> AppResult<()> {
    conn.execute(
        "UPDATE sessions SET course = ? WHERE name = ?",
        (course.unwrap_or(""), name),
    )
    .await?;
    Ok(())
}

pub async fn touch(conn: &Connection, name: &str) -> AppResult<()> {
    conn.execute(
        "UPDATE sessions SET last_updated = ? WHERE name = ?",
        (now_iso(), name),
    )
    .await?;
    Ok(())
}

pub async fn delete_session(conn: &Connection, name: &str) -> AppResult<bool> {
    let existed = session_exists(conn, name).await?;
    for sql in [
        "DELETE FROM graded_items WHERE session_name = ?",
        "DELETE FROM caches WHERE session_name = ?",
        "DELETE FROM questions WHERE session_name = ?",
        "DELETE FROM sessions WHERE name = ?",
    ] {
        conn.execute(sql, (name,)).await?;
    }
    Ok(existed)
}

async fn load_graded_items(conn: &Connection, name: &str, col: &str) -> AppResult<Vec<GradedItem>> {
    let mut items = Vec::new();
    let mut g = conn
        .query(
            "SELECT row_id, input_text, grade, timestamp FROM graded_items \
             WHERE session_name = ? AND output_col = ? ORDER BY timestamp, row_id",
            (name, col),
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
        external_graded_items: Vec::new(),
    };
    Ok((col, q))
}

const Q_COLS: &str =
    "output_col, question_name, input_column, exam_question, sample_answer, global_question_id, sampling_result";

pub async fn load_session(conn: &Connection, name: &str) -> AppResult<Option<Session>> {
    let meta = {
        let mut r = conn
            .query(
                "SELECT csv_file, id_columns, input_columns, output_columns, course, last_updated \
                 FROM sessions WHERE name = ?",
                (name,),
            )
            .await?;
        match r.next().await? {
            Some(row) => {
                let csv_file: String = row.get(0)?;
                let id_columns = parse_array(&row.get::<String>(1)?);
                let input_columns = parse_array(&row.get::<String>(2)?);
                let output_columns = parse_array(&row.get::<String>(3)?);
                let course = opt(row.get::<String>(4)?);
                let last_updated: String = row.get(5)?;
                (csv_file, id_columns, input_columns, output_columns, course, last_updated)
            }
            None => return Ok(None),
        }
    };

    // Question skeletons first (drop the Rows before per-question queries).
    let mut questions: HashMap<String, QuestionGrades> = HashMap::new();
    {
        let sql = format!("SELECT {Q_COLS} FROM questions WHERE session_name = ?");
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

    Ok(Some(Session {
        session_name: name.to_string(),
        csv_file: meta.0,
        id_columns: meta.1,
        input_columns: meta.2,
        output_columns: meta.3,
        course: meta.4,
        last_updated: meta.5,
        questions,
    }))
}

pub async fn load_question(
    conn: &Connection,
    name: &str,
    col: &str,
) -> AppResult<Option<QuestionGrades>> {
    let mut q = {
        let sql = format!("SELECT {Q_COLS} FROM questions WHERE session_name = ? AND output_col = ?");
        let mut r = conn.query(&sql, (name, col)).await?;
        match r.next().await? {
            Some(row) => question_from_row(&row)?.1,
            None => return Ok(None),
        }
    };
    q.graded_items = load_graded_items(conn, name, col).await?;
    hydrate_external(conn, &mut q, Some(name)).await?;
    Ok(Some(q))
}

pub async fn list_sessions(conn: &Connection, archived: bool) -> AppResult<Vec<SessionSummary>> {
    struct Row {
        name: String,
        csv: String,
        course: String,
        updated: String,
    }
    let mut tmp = Vec::new();
    {
        let mut r = conn
            .query(
                "SELECT name, csv_file, course, last_updated FROM sessions \
                 WHERE archived = ? ORDER BY last_updated DESC",
                (i64::from(archived),),
            )
            .await?;
        while let Some(row) = r.next().await? {
            tmp.push(Row {
                name: row.get(0)?,
                csv: row.get(1)?,
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
                    "SELECT COUNT(*) FROM questions WHERE session_name = ?",
                    (t.name.as_str(),),
                )
                .await?;
            c.next().await?.map(|r| r.get::<i64>(0)).transpose()?.unwrap_or(0) as usize
        };
        out.push(SessionSummary {
            session_name: t.name,
            csv_file: t.csv,
            course: opt(t.course),
            last_updated: t.updated,
            num_questions,
            archived,
        });
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Feature cache (Turso)
// ---------------------------------------------------------------------------

pub async fn has_feature_vector(
    conn: &Connection,
    name: &str,
    input_column: &str,
    row_id: &str,
) -> AppResult<bool> {
    let mut r = conn
        .query(
            "SELECT 1 FROM caches WHERE session_name = ? AND kind = 'features' AND input_column = ? AND row_id = ?",
            (name, input_column, row_id),
        )
        .await?;
    Ok(r.next().await?.is_some())
}

pub async fn put_feature_vector(
    conn: &Connection,
    name: &str,
    input_column: &str,
    row_id: &str,
    vector: &[f32],
) -> AppResult<()> {
    conn.execute(
        "DELETE FROM caches WHERE session_name = ? AND kind = 'features' AND input_column = ? AND row_id = ?",
        (name, input_column, row_id),
    )
    .await?;
    conn.execute(
        "INSERT INTO caches (session_name, kind, input_column, row_id, vector) \
         VALUES (?, 'features', ?, ?, ?)",
        (name, input_column, row_id, db::f32s_to_blob(vector)),
    )
    .await?;
    Ok(())
}

pub async fn load_feature_cache(
    conn: &Connection,
    name: &str,
    input_column: &str,
) -> AppResult<HashMap<String, Vec<f32>>> {
    let mut out = HashMap::new();
    let mut r = conn
        .query(
            "SELECT row_id, vector FROM caches WHERE session_name = ? AND kind = 'features' AND input_column = ?",
            (name, input_column),
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
// Cross-session graded pool (Turso)
// ---------------------------------------------------------------------------

pub async fn hydrate_external(
    conn: &Connection,
    question: &mut QuestionGrades,
    current_session: Option<&str>,
) -> AppResult<()> {
    let Some(gq) = question.global_question_id.clone() else {
        question.external_graded_items = Vec::new();
        return Ok(());
    };
    let current = current_session.unwrap_or("");
    let mut items = Vec::new();
    let mut r = conn
        .query(
            "SELECT source_session, row_id, input_text, grade, timestamp FROM graded_pool \
             WHERE global_question_id = ? AND source_session <> ?",
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

async fn sync_item_to_pool(
    conn: &Connection,
    gq_id: &str,
    source_session: &str,
    item: &GradedItem,
) -> AppResult<()> {
    let exists = {
        let mut r = conn
            .query(
                "SELECT 1 FROM graded_pool WHERE global_question_id = ? AND source_session = ? AND row_id = ?",
                (gq_id, source_session, item.row_id.as_str()),
            )
            .await?;
        r.next().await?.is_some()
    };
    if exists {
        return Ok(());
    }
    conn.execute(
        "INSERT INTO graded_pool (global_question_id, source_session, row_id, input_text, grade, timestamp) \
         VALUES (?, ?, ?, ?, ?, ?)",
        (
            gq_id,
            source_session,
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
// Legacy file import (JSON / jsonl -> Turso)
// ---------------------------------------------------------------------------

/// How to handle a session whose name already exists in the DB.
pub enum OnCollision<'a> {
    /// Idempotent startup import: skip if already present.
    Skip,
    /// Workspace import: keep both by suffixing with the workspace name.
    Suffix(&'a str),
}

/// Import one legacy session JSON value into the DB. Returns the stored name, or
/// `None` if skipped. `caches_dir`/`stem` locate the side-car `*.cache.json`.
pub async fn import_session_value(
    conn: &Connection,
    value: &Value,
    stem: &str,
    caches_dir: &Path,
    archived: bool,
    course_override: Option<&str>,
    collision: OnCollision<'_>,
) -> AppResult<Option<String>> {
    let orig = value
        .get("session_name")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| stem.to_string());

    let name = if session_exists(conn, &orig).await? {
        match collision {
            OnCollision::Skip => return Ok(None),
            OnCollision::Suffix(ws) => unique_name(conn, &orig, ws).await?,
        }
    } else {
        orig.clone()
    };

    let mut session: Session = serde_json::from_value(value.clone())?;
    session.session_name = name.clone();
    if let Some(c) = course_override {
        session.course = Some(c.to_string());
    }
    if session.last_updated.is_empty() {
        session.last_updated = now_iso();
    }
    insert_session(conn, &session).await?;
    if archived {
        set_archived(conn, &name, true).await?;
    }

    import_caches(conn, &name, stem, caches_dir, value).await?;
    Ok(Some(name))
}

async fn unique_name(conn: &Connection, orig: &str, ws: &str) -> AppResult<String> {
    let base = format!("{orig}_{ws}");
    if !session_exists(conn, &base).await? {
        return Ok(base);
    }
    let mut i = 2;
    loop {
        let candidate = format!("{base}_{i}");
        if !session_exists(conn, &candidate).await? {
            return Ok(candidate);
        }
        i += 1;
    }
}

async fn import_caches(
    conn: &Connection,
    name: &str,
    stem: &str,
    caches_dir: &Path,
    value: &Value,
) -> AppResult<()> {
    // Prefer the side-car cache file; fall back to caches embedded in the main file.
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
                put_feature_vector(conn, name, input_col, rid, &v).await?;
            }
        }
    }
    Ok(())
}

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
            let src = rec.get("source_session").and_then(|v| v.as_str()).unwrap_or("");
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

/// One-time, idempotent import of any legacy on-disk data into the DB.
pub async fn import_legacy_on_startup(conn: &Connection, config: &Config) -> AppResult<()> {
    // Oldest single-file session.
    let single = config.data_dir.join(".tentanator_session.json");
    if single.exists() {
        if let Ok(raw) = std::fs::read_to_string(&single) {
            if let Ok(value) = serde_json::from_str::<Value>(&raw) {
                let stem = single_file_name(&value);
                let _ = import_session_value(
                    conn,
                    &value,
                    &stem,
                    &config.sessions_dir(),
                    false,
                    None,
                    OnCollision::Skip,
                )
                .await?;
                let _ = std::fs::rename(
                    &single,
                    config.data_dir.join(".tentanator_session.json.backup"),
                );
            }
        }
    }

    import_session_dir(conn, &config.sessions_dir(), false).await?;
    import_session_dir(conn, &config.archive_dir(), true).await?;
    import_pool_dir(conn, &config.graded_pool_dir()).await?;
    Ok(())
}

fn single_file_name(value: &Value) -> String {
    let csv_base = value
        .get("csv_file")
        .and_then(|v| v.as_str())
        .and_then(|s| Path::new(s).file_stem().and_then(|x| x.to_str()))
        .unwrap_or("unknown");
    let ts: String = value
        .get("last_updated")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .chars()
        .take(19)
        .collect::<String>()
        .replace([':', '-'], "")
        .replace('T', "_");
    sanitize_name(&format!("{csv_base}_migrated_{ts}"))
}

pub async fn import_session_dir(conn: &Connection, dir: &Path, archived: bool) -> AppResult<()> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Ok(());
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
        import_session_value(conn, &value, stem, dir, archived, None, OnCollision::Skip).await?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Excel export (on disk)
// ---------------------------------------------------------------------------

pub fn export_to_excel(
    config: &Config,
    session: &Session,
    exam_rows: &[HashMap<String, String>],
    columns: &[String],
) -> AppResult<String> {
    use rust_xlsxwriter::Workbook;

    std::fs::create_dir_all(config.graded_dir())?;
    let base = Path::new(&session.csv_file)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("graded")
        .to_string();
    let out_path = config.graded_dir().join(format!("{base}.xlsx"));

    let mut grades_by_row: HashMap<String, HashMap<String, String>> = HashMap::new();
    for (col, q) in &session.questions {
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
        let rid = crate::domain::row_id(row, &session.id_columns);
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
    workbook
        .save(&out_path)
        .map_err(|e| AppError::Other(anyhow::anyhow!(e)))?;
    Ok(out_path.to_string_lossy().to_string())
}

fn format_total(total: f64) -> String {
    if total.fract() == 0.0 && total.is_finite() {
        format!("{}", total as i64)
    } else {
        format!("{total}")
    }
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

    #[tokio::test]
    async fn create_grade_load_roundtrip() {
        let conn = mem_conn().await;
        let mut session = Session {
            session_name: "s1".into(),
            csv_file: "e.csv".into(),
            id_columns: vec!["id".into()],
            input_columns: vec!["ans".into()],
            output_columns: vec!["g".into()],
            course: Some("CS101".into()),
            last_updated: now_iso(),
            questions: HashMap::new(),
        };
        session.ensure_question("g");
        insert_session(&conn, &session).await.unwrap();

        let item = GradedItem {
            row_id: "r1".into(),
            input_text: "hello".into(),
            grade: "2+1.5".into(),
            timestamp: now_iso(),
        };
        put_graded_item(&conn, "s1", "g", &item).await.unwrap();

        let loaded = load_session(&conn, "s1").await.unwrap().unwrap();
        assert_eq!(loaded.course.as_deref(), Some("CS101"));
        let q = loaded.questions.get("g").unwrap();
        assert_eq!(q.graded_items.len(), 1);
        assert_eq!(q.graded_items[0].grade, "2+1.5");

        // archive flips listing.
        assert!(list_sessions(&conn, false).await.unwrap().iter().any(|s| s.session_name == "s1"));
        set_archived(&conn, "s1", true).await.unwrap();
        assert!(list_sessions(&conn, false).await.unwrap().is_empty());
        assert_eq!(list_sessions(&conn, true).await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn pool_hydrates_across_sessions() {
        let conn = mem_conn().await;
        for name in ["sa", "sb"] {
            let mut s = Session {
                session_name: name.into(),
                csv_file: "e.csv".into(),
                id_columns: vec!["id".into()],
                input_columns: vec!["ans".into()],
                output_columns: vec!["g".into()],
                course: None,
                last_updated: now_iso(),
                questions: HashMap::new(),
            };
            let q = s.ensure_question("g");
            q.global_question_id = Some("gq7".into());
            insert_session(&conn, &s).await.unwrap();
        }
        // Grade in sa -> should appear as external for sb.
        put_graded_item(
            &conn,
            "sa",
            "g",
            &GradedItem { row_id: "r1".into(), input_text: "x".into(), grade: "5".into(), timestamp: now_iso() },
        )
        .await
        .unwrap();

        let qb = load_question(&conn, "sb", "g").await.unwrap().unwrap();
        assert_eq!(qb.external_graded_items.len(), 1);
        assert_eq!(qb.external_graded_items[0].row_id, "sa::r1");
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
