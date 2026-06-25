//! Workspace compatibility.
//!
//! The legacy `workspace.py` stores each workspace as a folder under
//! `workspaces/<name>/` containing the same subdirs (`.tentanator_sessions/`,
//! `exams/`, `graded_exams/`, `global_bank/`, ...). The currently "loaded"
//! workspace has its contents moved to the data-dir root, and its name recorded
//! in `.current_workspace`.
//!
//! The backend reads the root (the active workspace) by default. Passing
//! `?workspace=<name>` to a data endpoint reads `workspaces/<name>/` instead, so
//! inactive workspaces can be browsed without the legacy move dance.

use serde::Serialize;

use crate::config::Config;

#[derive(Clone, Debug, Serialize)]
pub struct WorkspaceInfo {
    pub name: String,
    pub sessions: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct WorkspaceListing {
    /// Name recorded in `.current_workspace`, if any (the data at the root).
    pub current: Option<String>,
    pub workspaces: Vec<WorkspaceInfo>,
}

fn count_sessions(dir: &std::path::Path) -> usize {
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

pub fn list_workspaces(config: &Config) -> WorkspaceListing {
    let current = std::fs::read_to_string(config.data_dir.join(".current_workspace"))
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    let mut workspaces = Vec::new();
    if let Ok(entries) = std::fs::read_dir(config.data_dir.join("workspaces")) {
        for e in entries.flatten() {
            if e.path().is_dir() {
                workspaces.push(WorkspaceInfo {
                    name: e.file_name().to_string_lossy().to_string(),
                    sessions: count_sessions(&e.path()),
                });
            }
        }
    }
    workspaces.sort_by(|a, b| a.name.cmp(&b.name));
    WorkspaceListing { current, workspaces }
}

/// Resolve the effective data directory for an optional workspace selection.
/// Returns the workspace folder if it exists, otherwise the root (active).
pub fn resolve_data_dir(config: &Config, workspace: Option<&str>) -> std::path::PathBuf {
    match workspace {
        Some(w) if !w.is_empty() => {
            let dir = config.data_dir.join("workspaces").join(w);
            if dir.is_dir() {
                dir
            } else {
                config.data_dir.clone()
            }
        }
        _ => config.data_dir.clone(),
    }
}
