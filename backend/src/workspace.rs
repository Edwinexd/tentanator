//! Legacy workspace import.
//!
//! The old `workspace.py` stored each project as a folder under
//! `workspaces/<name>/` and switched between them by moving the contents in and
//! out of the data root. That was a band-aid for the lack of a real session
//! model. The new system is a single flat store; sessions are grouped by an
//! optional `course` tag instead of by location.
//!
//! This module only exists to import that old data: it lists the leftover
//! `workspaces/<name>/` folders and copies their sessions, caches, exams and
//! graded pool into the flat store, tagging the imported sessions with the
//! workspace name as their `course`.

use std::collections::HashSet;
use std::path::Path;

use serde::Serialize;
use serde_json::Value;

use crate::config::Config;
use crate::error::{AppError, AppResult};

#[derive(Clone, Debug, Serialize)]
pub struct WorkspaceInfo {
    pub name: String,
    pub sessions: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct ImportResult {
    pub imported_sessions: Vec<String>,
    pub imported_exams: usize,
    pub skipped_exams: usize,
}

fn count_sessions(dir: &Path) -> usize {
    let sessions = dir.join(".tentanator_sessions");
    let Ok(entries) = std::fs::read_dir(&sessions) else {
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

/// List the leftover legacy workspaces available to import.
pub fn list_importable(config: &Config) -> Vec<WorkspaceInfo> {
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir(config.data_dir.join("workspaces")) {
        for e in entries.flatten() {
            if e.path().is_dir() {
                out.push(WorkspaceInfo {
                    name: e.file_name().to_string_lossy().to_string(),
                    sessions: count_sessions(&e.path()),
                });
            }
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// Import a legacy `workspaces/<name>/` folder into the flat store. Sessions are
/// tagged with `course = <name>` (unless they already carry a course). Existing
/// exams and session names are not overwritten (sessions get a suffix).
pub fn import_workspace(config: &Config, name: &str) -> AppResult<ImportResult> {
    let ws = config.data_dir.join("workspaces").join(name);
    if !ws.is_dir() {
        return Err(AppError::NotFound(format!("workspace '{name}' not found")));
    }

    // -- exams (never overwrite an existing exam) --------------------------
    let mut imported_exams = 0usize;
    let mut skipped_exams = 0usize;
    let dst_exams = config.exams_dir();
    std::fs::create_dir_all(&dst_exams)?;
    if let Ok(entries) = std::fs::read_dir(ws.join("exams")) {
        for e in entries.flatten() {
            if e.path().is_file() {
                let dst = dst_exams.join(e.file_name());
                if dst.exists() {
                    skipped_exams += 1;
                } else {
                    std::fs::copy(e.path(), &dst)?;
                    imported_exams += 1;
                }
            }
        }
    }

    // -- sessions (+ caches), tagged with the course -----------------------
    let dst_sessions = config.sessions_dir();
    std::fs::create_dir_all(&dst_sessions)?;
    let src_sessions = ws.join(".tentanator_sessions");
    let mut imported_sessions = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&src_sessions) {
        for e in entries.flatten() {
            let fname = e.file_name().to_string_lossy().to_string();
            if !fname.ends_with(".json") || fname.ends_with(".cache.json") {
                continue;
            }
            let orig = fname.trim_end_matches(".json").to_string();
            let Ok(raw) = std::fs::read_to_string(e.path()) else {
                continue;
            };
            let Ok(mut value) = serde_json::from_str::<Value>(&raw) else {
                continue;
            };

            let target = unique_session_name(&dst_sessions, &orig, name);
            if let Some(obj) = value.as_object_mut() {
                obj.insert("session_name".into(), Value::String(target.clone()));
                let has_course = obj
                    .get("course")
                    .and_then(|c| c.as_str())
                    .map(|s| !s.is_empty())
                    .unwrap_or(false);
                if !has_course {
                    obj.insert("course".into(), Value::String(name.to_string()));
                }
            }
            let Ok(bytes) = serde_json::to_vec_pretty(&value) else {
                continue;
            };
            if std::fs::write(dst_sessions.join(format!("{target}.json")), bytes).is_ok() {
                let src_cache = src_sessions.join(format!("{orig}.cache.json"));
                if src_cache.exists() {
                    let _ = std::fs::copy(
                        &src_cache,
                        dst_sessions.join(format!("{target}.cache.json")),
                    );
                }
                imported_sessions.push(target);
            }
        }
    }

    // -- cross-session graded pool (merge, dedup) --------------------------
    let src_pool = ws.join("global_bank").join("graded_pool");
    if src_pool.is_dir() {
        let dst_pool = config.graded_pool_dir();
        let _ = std::fs::create_dir_all(&dst_pool);
        if let Ok(entries) = std::fs::read_dir(&src_pool) {
            for e in entries.flatten() {
                let fname = e.file_name().to_string_lossy().to_string();
                if fname.ends_with(".jsonl") {
                    merge_pool_file(&e.path(), &dst_pool.join(&fname));
                }
            }
        }
    }

    Ok(ImportResult {
        imported_sessions,
        imported_exams,
        skipped_exams,
    })
}

fn unique_session_name(dst_sessions: &Path, orig: &str, ws_name: &str) -> String {
    if !dst_sessions.join(format!("{orig}.json")).exists() {
        return orig.to_string();
    }
    let base = format!("{orig}_{ws_name}");
    if !dst_sessions.join(format!("{base}.json")).exists() {
        return base;
    }
    let mut i = 2;
    loop {
        let candidate = format!("{base}_{i}");
        if !dst_sessions.join(format!("{candidate}.json")).exists() {
            return candidate;
        }
        i += 1;
    }
}

fn pool_keys(path: &Path) -> HashSet<(String, String)> {
    let Ok(content) = std::fs::read_to_string(path) else {
        return HashSet::new();
    };
    content
        .lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .map(|v| {
            (
                v.get("source_session").and_then(|x| x.as_str()).unwrap_or("").to_string(),
                v.get("row_id").and_then(|x| x.as_str()).unwrap_or("").to_string(),
            )
        })
        .collect()
}

fn merge_pool_file(src: &Path, dst: &Path) {
    use std::io::Write;
    let mut existing = pool_keys(dst);
    let Ok(src_content) = std::fs::read_to_string(src) else {
        return;
    };
    let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(dst) else {
        return;
    };
    for line in src_content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Ok(v) = serde_json::from_str::<Value>(trimmed) {
            let key = (
                v.get("source_session").and_then(|x| x.as_str()).unwrap_or("").to_string(),
                v.get("row_id").and_then(|x| x.as_str()).unwrap_or("").to_string(),
            );
            if existing.insert(key) {
                let _ = writeln!(f, "{trimmed}");
            }
        }
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
        let dir = std::env::temp_dir().join(format!("tt-ws-test-{}-{}", std::process::id(), n));
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
    fn imports_workspace_and_tags_course() {
        let config = tmp_config();
        let ws = config.data_dir.join("workspaces").join("CS101");
        std::fs::create_dir_all(ws.join("exams")).unwrap();
        std::fs::create_dir_all(ws.join(".tentanator_sessions")).unwrap();
        std::fs::write(ws.join("exams").join("e.csv"), "id\n1\n").unwrap();
        std::fs::write(
            ws.join(".tentanator_sessions").join("s.json"),
            r#"{"session_name":"s","csv_file":"e.csv","id_columns":["id"],"input_columns":["a"],"output_columns":["g"],"questions":{}}"#,
        )
        .unwrap();

        let result = import_workspace(&config, "CS101").expect("import ok");
        assert_eq!(result.imported_sessions, vec!["s".to_string()]);
        assert_eq!(result.imported_exams, 1);

        // Imported session is in the flat store and tagged with the course.
        let imported =
            std::fs::read_to_string(config.sessions_dir().join("s.json")).unwrap();
        let value: Value = serde_json::from_str(&imported).unwrap();
        assert_eq!(value.get("course").unwrap().as_str().unwrap(), "CS101");
        assert!(config.exams_dir().join("e.csv").exists());
        let _ = std::fs::remove_dir_all(&config.data_dir);
    }
}
