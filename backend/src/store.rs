//! Filesystem persistence + exam I/O. Mirrors the legacy on-disk layout so
//! existing sessions, caches, pools and exam files remain compatible.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use calamine::{open_workbook_auto, Data, Reader};
use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::domain::{Cache, GradedItem, QuestionGrades, Session, SessionSummary};
use crate::error::{AppError, AppResult};

// ---------------------------------------------------------------------------
// Timestamps & name sanitization
// ---------------------------------------------------------------------------

pub fn now_iso() -> String {
    chrono::Local::now()
        .format("%Y-%m-%dT%H:%M:%S%.6f")
        .to_string()
}

pub fn timestamp_compact() -> String {
    chrono::Local::now().format("%Y%m%d_%H%M%S").to_string()
}

pub fn sanitize_name(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
        .collect()
}

// ---------------------------------------------------------------------------
// Exam files
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

/// Resolve an exam file under exams/, trying alternate extensions.
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

fn is_xlsx(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()).map(|s| s.to_lowercase()),
        Some(ref e) if e == "xlsx" || e == "xls"
    )
}

// ---------------------------------------------------------------------------
// Sessions
// ---------------------------------------------------------------------------

#[derive(Default, Serialize, Deserialize)]
struct CacheFile {
    #[serde(default)]
    embeddings_cache: Cache,
    #[serde(default)]
    features_cache: Cache,
}

fn base_dir(config: &Config, archived: bool) -> PathBuf {
    if archived {
        config.archive_dir()
    } else {
        config.sessions_dir()
    }
}

fn session_path(config: &Config, name: &str, archived: bool) -> PathBuf {
    base_dir(config, archived).join(format!("{name}.json"))
}

fn cache_path(config: &Config, name: &str, archived: bool) -> PathBuf {
    base_dir(config, archived).join(format!("{name}.cache.json"))
}

/// Migrate the pre-`.tentanator_sessions/` single-file session
/// (`.tentanator_session.json`) into the directory layout. Best-effort, mirrors
/// the legacy Python migration so the oldest sessions still import.
pub fn migrate_legacy_single_session(config: &Config) {
    let old = config.data_dir.join(".tentanator_session.json");
    if !old.exists() {
        return;
    }
    let Ok(raw) = std::fs::read_to_string(&old) else {
        return;
    };
    let Ok(mut value) = serde_json::from_str::<serde_json::Value>(&raw) else {
        return;
    };

    let csv_base = value
        .get("csv_file")
        .and_then(|v| v.as_str())
        .and_then(|s| Path::new(s).file_stem().and_then(|x| x.to_str()))
        .unwrap_or("unknown")
        .to_string();
    let ts: String = value
        .get("last_updated")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .chars()
        .take(19)
        .collect::<String>()
        .replace([':', '-'], "")
        .replace('T', "_");
    let name = sanitize_name(&format!("{csv_base}_migrated_{ts}"));

    if std::fs::create_dir_all(config.sessions_dir()).is_err() {
        return;
    }
    let dest = session_path(config, &name, false);
    if dest.exists() {
        return;
    }
    if let Some(obj) = value.as_object_mut() {
        obj.insert("session_name".into(), serde_json::Value::String(name.clone()));
    }
    if let Ok(bytes) = serde_json::to_vec_pretty(&value) {
        if std::fs::write(&dest, bytes).is_ok() {
            let _ = std::fs::rename(&old, config.data_dir.join(".tentanator_session.json.backup"));
        }
    }
}

pub fn save_session(config: &Config, session: &mut Session, name: &str) -> AppResult<String> {
    let name = sanitize_name(name);
    session.last_updated = now_iso();
    session.session_name = name.clone();

    std::fs::create_dir_all(config.sessions_dir())?;

    // Caches go to a side file.
    let cache = CacheFile {
        embeddings_cache: session.embeddings_cache.clone(),
        features_cache: session.features_cache.clone(),
    };
    std::fs::write(
        cache_path(config, &name, false),
        serde_json::to_vec(&cache)?,
    )?;

    // Session file (caches + external items are skipped by serde).
    std::fs::write(
        session_path(config, &name, false),
        serde_json::to_vec_pretty(session)?,
    )?;

    // Push any new graded items into the cross-session pool.
    for q in session.questions.values() {
        let _ = sync_question_to_pool(config, q, &name);
    }
    Ok(name)
}

pub fn load_session(config: &Config, name: &str, archived: bool) -> AppResult<Session> {
    let path = session_path(config, name, archived);
    if !path.exists() {
        return Err(AppError::NotFound(format!("session '{name}' not found")));
    }
    let raw = std::fs::read_to_string(&path)?;
    let value: serde_json::Value = serde_json::from_str(&raw)?;

    let mut session: Session = serde_json::from_value(value.clone())?;
    session.session_name = name.to_string();

    // Caches: prefer side file, else fall back to embedded (old format).
    let cf_path = cache_path(config, name, archived);
    if cf_path.exists() {
        if let Ok(cf) = serde_json::from_str::<CacheFile>(&std::fs::read_to_string(&cf_path)?) {
            session.embeddings_cache = cf.embeddings_cache;
            session.features_cache = cf.features_cache;
        }
    } else {
        if let Some(c) = value.get("embeddings_cache") {
            session.embeddings_cache = serde_json::from_value(c.clone()).unwrap_or_default();
        }
        if let Some(c) = value.get("features_cache") {
            session.features_cache = serde_json::from_value(c.clone()).unwrap_or_default();
        }
    }

    // Hydrate cross-session pooled examples for linked questions.
    let session_name = session.session_name.clone();
    for q in session.questions.values_mut() {
        hydrate_external_graded_items(config, q, Some(&session_name));
    }
    Ok(session)
}

pub fn list_sessions(config: &Config, archived: bool) -> Vec<SessionSummary> {
    let dir = base_dir(config, archived);
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return out;
    };
    for e in entries.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        if !name.ends_with(".json") || name.ends_with(".cache.json") {
            continue;
        }
        let Ok(raw) = std::fs::read_to_string(e.path()) else {
            continue;
        };
        let Ok(value) = serde_json::from_str::<serde_json::Value>(&raw) else {
            continue;
        };
        let csv_file = value
            .get("csv_file")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        // Only surface sessions whose exam file still exists (legacy behavior).
        if resolve_exam_path(config, &csv_file).is_none() {
            continue;
        }
        let session_name = name.trim_end_matches(".json").to_string();
        out.push(SessionSummary {
            session_name,
            csv_file,
            course: value
                .get("course")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            last_updated: value
                .get("last_updated")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            num_questions: value
                .get("questions")
                .and_then(|v| v.as_object())
                .map(|o| o.len())
                .unwrap_or(0),
            archived,
        });
    }
    out.sort_by(|a, b| b.last_updated.cmp(&a.last_updated));
    out
}

pub fn delete_session(config: &Config, name: &str) -> AppResult<()> {
    let mut removed = false;
    for archived in [false, true] {
        let p = session_path(config, name, archived);
        if p.exists() {
            std::fs::remove_file(&p)?;
            removed = true;
        }
        let c = cache_path(config, name, archived);
        if c.exists() {
            std::fs::remove_file(&c)?;
        }
    }
    if removed {
        Ok(())
    } else {
        Err(AppError::NotFound(format!("session '{name}' not found")))
    }
}

pub fn set_archived(config: &Config, name: &str, archived: bool) -> AppResult<()> {
    // Move from the source dir to the destination dir.
    let (from, to) = (!archived, archived);
    let src = session_path(config, name, from);
    if !src.exists() {
        return Err(AppError::NotFound(format!("session '{name}' not found")));
    }
    std::fs::create_dir_all(base_dir(config, to))?;
    std::fs::rename(&src, session_path(config, name, to))?;
    let src_cache = cache_path(config, name, from);
    if src_cache.exists() {
        std::fs::rename(&src_cache, cache_path(config, name, to))?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Cross-session graded pool
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
struct PoolRec {
    row_id: String,
    input_text: String,
    grade: String,
    #[serde(default)]
    timestamp: String,
    #[serde(default)]
    source_session: String,
}

fn pool_file(config: &Config, gq_id: &str) -> PathBuf {
    config.graded_pool_dir().join(format!("{gq_id}.jsonl"))
}

fn read_pool_lines(config: &Config, gq_id: &str) -> Vec<PoolRec> {
    let path = pool_file(config, gq_id);
    let Ok(content) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };
    content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str::<PoolRec>(l).ok())
        .collect()
}

pub fn hydrate_external_graded_items(
    config: &Config,
    question: &mut QuestionGrades,
    current_session: Option<&str>,
) {
    let Some(gq_id) = question.global_question_id.clone() else {
        question.external_graded_items = Vec::new();
        return;
    };
    let mut items = Vec::new();
    for rec in read_pool_lines(config, &gq_id) {
        if current_session.is_some_and(|s| s == rec.source_session) {
            continue;
        }
        let src = if rec.source_session.is_empty() {
            "external".to_string()
        } else {
            rec.source_session.clone()
        };
        items.push(GradedItem {
            row_id: format!("{src}::{}", rec.row_id),
            input_text: rec.input_text,
            grade: rec.grade,
            timestamp: rec.timestamp,
        });
    }
    question.external_graded_items = items;
}

pub fn sync_question_to_pool(config: &Config, question: &QuestionGrades, session_name: &str) -> usize {
    let Some(gq_id) = question.global_question_id.clone() else {
        return 0;
    };
    if session_name.is_empty() {
        return 0;
    }
    if std::fs::create_dir_all(config.graded_pool_dir()).is_err() {
        return 0;
    }
    let mut existing: HashSet<(String, String)> = read_pool_lines(config, &gq_id)
        .into_iter()
        .map(|r| (r.source_session, r.row_id))
        .collect();

    use std::io::Write;
    let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(pool_file(config, &gq_id))
    else {
        return 0;
    };
    let mut appended = 0;
    for item in &question.graded_items {
        let key = (session_name.to_string(), item.row_id.clone());
        if existing.contains(&key) {
            continue;
        }
        let rec = PoolRec {
            row_id: item.row_id.clone(),
            input_text: item.input_text.clone(),
            grade: item.grade.clone(),
            timestamp: item.timestamp.clone(),
            source_session: session_name.to_string(),
        };
        if let Ok(line) = serde_json::to_string(&rec) {
            if writeln!(f, "{line}").is_ok() {
                existing.insert(key);
                appended += 1;
            }
        }
    }
    appended
}

// ---------------------------------------------------------------------------
// Excel export
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

    // row_id -> { output_col -> grade (numeric total when parseable) }
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

/// Format a numeric total like Python's `f"{total:g}"` (trim trailing zeros).
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
    use crate::config::Config;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static CTR: AtomicUsize = AtomicUsize::new(0);

    fn tmp_config() -> Config {
        let n = CTR.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("tt-store-test-{}-{}", std::process::id(), n));
        let _ = std::fs::create_dir_all(&dir);
        Config {
            data_dir: dir,
            openai_api_key: String::new(),
            openai_base_url: "http://localhost".into(),
            embedding_model: "x".into(),
            cerebras_api_key: String::new(),
            cerebras_base_url: "http://localhost".into(),
            cerebras_model: "x".into(),
            cerebras_reasoning_effort: "high".into(),
            cerebras_summary_model: "x".into(),
            bind_addr: "127.0.0.1:0".into(),
        }
    }

    #[test]
    fn loads_legacy_session_with_embedded_caches_and_extra_fields() {
        let config = tmp_config();
        std::fs::create_dir_all(config.sessions_dir()).unwrap();
        // Legacy quirks: caches embedded in the main file (no side-car) and a
        // per-item `embedding` field that the current model no longer stores.
        let legacy = r#"{
          "session_name": "old",
          "csv_file": "exam.xlsx",
          "id_columns": ["sid"],
          "input_columns": ["ans"],
          "output_columns": ["grade"],
          "last_updated": "2024-01-02T03:04:05",
          "questions": {
            "grade": {
              "question_name": "grade",
              "input_column": "ans",
              "exam_question": "Q?",
              "graded_items": [
                {"row_id":"s1","input_text":"hello","grade":"2+1.5","timestamp":"t","embedding":[0.1,0.2]}
              ],
              "sampling_result": {"algorithm":"maximin","selected_ids":["s1"],"quality_score":0.0,"num_samples":1,"timestamp":"t"}
            }
          },
          "embeddings_cache": {"ans": {"s1": [0.1, 0.2]}},
          "features_cache": {"ans": {"s1": [0.3, 0.4]}}
        }"#;
        std::fs::write(config.sessions_dir().join("old.json"), legacy).unwrap();

        let session = load_session(&config, "old", false).expect("legacy session loads");
        assert_eq!(session.csv_file, "exam.xlsx");
        let q = session.questions.get("grade").unwrap();
        assert_eq!(q.graded_items.len(), 1);
        assert_eq!(q.graded_items[0].grade, "2+1.5");
        assert_eq!(q.sampling_result.as_ref().unwrap().algorithm, "maximin");
        // Embedded caches hydrate when there is no side-car cache file.
        assert!(session
            .features_cache
            .get("ans")
            .and_then(|m| m.get("s1"))
            .is_some());
        let _ = std::fs::remove_dir_all(&config.data_dir);
    }

    #[test]
    fn migrates_single_file_session() {
        let config = tmp_config();
        let old = r#"{"csv_file":"exam.csv","id_columns":["sid"],"input_columns":["ans"],"output_columns":["grade"],"last_updated":"2024-05-06T07:08:09","questions":{}}"#;
        std::fs::write(config.data_dir.join(".tentanator_session.json"), old).unwrap();
        migrate_legacy_single_session(&config);
        assert!(config
            .data_dir
            .join(".tentanator_session.json.backup")
            .exists());
        let migrated: Vec<String> = std::fs::read_dir(config.sessions_dir())
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.ends_with(".json"))
            .collect();
        assert_eq!(migrated.len(), 1);
        assert!(migrated[0].contains("migrated"));
        let _ = std::fs::remove_dir_all(&config.data_dir);
    }
}
