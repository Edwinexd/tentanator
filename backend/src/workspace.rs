//! On-demand legacy import (old Python-app data -> new exam-centric store).
//!
//! The pre-Rust app stored each project as a folder under `workspaces/<name>/`.
//! This module lists those leftover folders and, on request, imports their old
//! session JSON files as exams (one `Exam` + a `default` session each, tagged
//! `course = <name>`), copies their exam files into `exams/`, and merges their
//! graded pool. It is NOT run on startup; trigger it via
//! `/api/legacy-workspaces/{name}/import`. Existing exams/files are never
//! overwritten (name collisions are suffixed with the workspace name).

use std::path::{Path, PathBuf};

use serde::Serialize;
use serde_json::Value;
use ts_rs::TS;
use turso::Connection;

use crate::config::Config;
use crate::error::{AppError, AppResult};
use crate::store;

#[derive(Clone, Debug, Serialize, TS)]
#[ts(export, export_to = "../../web/src/lib/generated/")]
pub struct WorkspaceInfo {
    pub name: String,
    /// Number of importable legacy session files in the workspace.
    pub exams: usize,
}

#[derive(Clone, Debug, Serialize, TS)]
#[ts(export, export_to = "../../web/src/lib/generated/")]
pub struct ImportResult {
    /// Names of the exams created in the new store.
    pub imported_exams: Vec<String>,
    pub imported_files: usize,
    pub skipped_files: usize,
}

/// Collect importable legacy session files in a workspace's
/// `.tentanator_sessions/`, including its `archive/` subfolder (archived
/// sessions are still real exams the user may want back). Cache sidecars
/// (`<stem>.cache.json`) are excluded; they're picked up by their parent dir.
fn session_files(dir: &Path) -> Vec<PathBuf> {
    let sessions = dir.join(".tentanator_sessions");
    let mut out = Vec::new();
    for base in [sessions.clone(), sessions.join("archive")] {
        let Ok(entries) = std::fs::read_dir(&base) else {
            continue;
        };
        for e in entries.flatten() {
            let n = e.file_name().to_string_lossy().to_string();
            if n.ends_with(".json") && !n.ends_with(".cache.json") {
                out.push(e.path());
            }
        }
    }
    out
}

/// List the leftover legacy workspaces available to import.
pub fn list_importable(config: &Config) -> Vec<WorkspaceInfo> {
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir(config.data_dir.join("workspaces")) {
        for e in entries.flatten() {
            if e.path().is_dir() {
                out.push(WorkspaceInfo {
                    name: e.file_name().to_string_lossy().to_string(),
                    exams: session_files(&e.path()).len(),
                });
            }
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// Import a legacy `workspaces/<name>/` folder into the DB in the new format.
/// Each old session becomes an exam (tagged `course = <name>`, suffixed on
/// collision); exam files are copied into `exams/` (never overwriting); the
/// graded pool is merged.
pub async fn import_workspace(
    conn: &Connection,
    config: &Config,
    name: &str,
) -> AppResult<ImportResult> {
    let ws = config.data_dir.join("workspaces").join(name);
    if !ws.is_dir() {
        return Err(AppError::NotFound(format!("workspace '{name}' not found")));
    }

    // Exam files (never overwrite an existing one).
    let mut imported_files = 0usize;
    let mut skipped_files = 0usize;
    let dst_exams = config.exams_dir();
    std::fs::create_dir_all(&dst_exams)?;
    if let Ok(entries) = std::fs::read_dir(ws.join("exams")) {
        for e in entries.flatten() {
            if e.path().is_file() {
                let dst = dst_exams.join(e.file_name());
                if dst.exists() {
                    skipped_files += 1;
                } else {
                    std::fs::copy(e.path(), &dst)?;
                    imported_files += 1;
                }
            }
        }
    }

    // Sessions (+ caches) into the DB as exams, tagged with the course.
    // Includes archived sessions (in `.tentanator_sessions/archive/`).
    let mut imported_exams = Vec::new();
    for path in session_files(&ws) {
        let fname = path.file_name().unwrap_or_default().to_string_lossy().to_string();
        let stem = fname.trim_end_matches(".json");
        // Cache sidecars sit next to the session file (archived caches live in
        // `archive/` alongside the archived session).
        let caches_dir = path.parent().unwrap_or(&ws);
        let Ok(raw) = std::fs::read_to_string(&path) else {
            continue;
        };
        let Ok(value) = serde_json::from_str::<Value>(&raw) else {
            continue;
        };
        let stored =
            store::import_legacy_session(conn, &value, stem, caches_dir, Some(name), name).await?;
        imported_exams.push(stored);
    }

    // Cross-exam graded pool.
    store::import_pool_dir(conn, &ws.join("global_bank").join("graded_pool")).await?;

    Ok(ImportResult {
        imported_exams,
        imported_files,
        skipped_files,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::db;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static CTR: AtomicUsize = AtomicUsize::new(0);

    fn tmp_config() -> Config {
        let n = CTR.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("tt-ws-db-{}-{}", std::process::id(), n));
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
            renderer_url: String::new(),
        }
    }

    #[tokio::test]
    async fn import_tags_course_and_creates_exam() {
        let config = tmp_config();
        let ws = config.data_dir.join("workspaces").join("CS101");
        std::fs::create_dir_all(ws.join("exams")).unwrap();
        std::fs::create_dir_all(ws.join(".tentanator_sessions")).unwrap();
        std::fs::write(ws.join("exams").join("e.xlsx"), b"x").unwrap();
        std::fs::write(
            ws.join(".tentanator_sessions").join("s.json"),
            r#"{"session_name":"s","csv_file":"e.xlsx","id_columns":["id"],"input_columns":["a"],"output_columns":["g"],"questions":{}}"#,
        )
        .unwrap();

        let database = db::open(config.data_dir.join("t.db").to_str().unwrap()).await.unwrap();
        let conn = database.connect().unwrap();
        db::init_schema(&conn).await.unwrap();

        let result = import_workspace(&conn, &config, "CS101").await.unwrap();
        assert_eq!(result.imported_exams, vec!["s".to_string()]);
        assert_eq!(result.imported_files, 1);

        let loaded = store::load_exam(&conn, "s").await.unwrap().unwrap();
        assert_eq!(loaded.course.as_deref(), Some("CS101"));
        assert_eq!(loaded.exam_file, "e.xlsx");
        // A default grading session is created for the imported exam.
        let sessions = store::list_sessions(&conn, "s").await.unwrap();
        assert!(sessions.iter().any(|x| x.name == "default"));
        assert!(config.exams_dir().join("e.xlsx").exists());
    }

    #[tokio::test]
    async fn import_includes_archived_sessions() {
        let config = tmp_config();
        let ws = config.data_dir.join("workspaces").join("PVT15");
        let archive = ws.join(".tentanator_sessions").join("archive");
        std::fs::create_dir_all(&archive).unwrap();
        std::fs::write(
            ws.join(".tentanator_sessions").join("active.json"),
            r#"{"session_name":"active","csv_file":"e.xlsx","id_columns":["id"],"input_columns":["a"],"output_columns":["g"],"questions":{}}"#,
        )
        .unwrap();
        std::fs::write(
            archive.join("reexam.json"),
            r#"{"session_name":"reexam","csv_file":"r.xlsx","id_columns":["id"],"input_columns":["a"],"output_columns":["g"],"questions":{}}"#,
        )
        .unwrap();

        // Both the active and archived session are counted as importable.
        let infos = list_importable(&config);
        let pvt = infos.iter().find(|w| w.name == "PVT15").unwrap();
        assert_eq!(pvt.exams, 2);

        let database = db::open(config.data_dir.join("t.db").to_str().unwrap()).await.unwrap();
        let conn = database.connect().unwrap();
        db::init_schema(&conn).await.unwrap();

        let result = import_workspace(&conn, &config, "PVT15").await.unwrap();
        assert_eq!(result.imported_exams.len(), 2);
        assert!(result.imported_exams.contains(&"reexam".to_string()));
        assert!(store::load_exam(&conn, "reexam").await.unwrap().is_some());
    }
}
