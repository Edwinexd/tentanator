//! Background results-PDF jobs.
//!
//! A results PDF compiles LaTeX per student and merges scan covers, which takes
//! minutes for a few hundred students. Rather than holding one HTTP request open
//! that long, `POST …/export/results-pdf` starts a job and returns at once; the
//! render runs on its own task, so a client may leave and poll
//! `GET …/export/results-pdf` later. There is at most one job per exam (the
//! latest), kept in memory: a backend restart forgets it, but a finished PDF
//! stays in `graded_exams/`.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::Mutex;
use ts_rs::TS;

use crate::store::now_iso;

/// A render takes minutes for a few hundred students; allow it plenty.
const RENDER_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30 * 60);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, TS)]
#[serde(rename_all = "lowercase")]
#[ts(export, export_to = "../../web/src/lib/generated/")]
pub enum PdfJobStatus {
    Running,
    Done,
    Failed,
}

/// The latest results-PDF job for an exam. On success `filename` names the PDF
/// in `graded_exams/` (download via `GET /api/graded/{filename}`).
#[derive(Clone, Debug, Serialize, Deserialize, TS)]
#[ts(export, export_to = "../../web/src/lib/generated/")]
pub struct PdfJob {
    pub exam: String,
    pub status: PdfJobStatus,
    pub scanned_pdf: Option<String>,
    pub started_at: String,
    pub finished_at: Option<String>,
    pub filename: Option<String>,
    pub students: Option<u32>,
    /// Student ids with no matching cover page in the scan.
    pub covers_missing: Vec<String>,
    pub error: Option<String>,
}

/// Latest job per exam name.
pub type PdfJobs = Arc<Mutex<HashMap<String, PdfJob>>>;

/// Start a render for `exam` unless one is already running, in which case that
/// job is returned unchanged (so a double click or a second client is harmless).
pub async fn start(
    jobs: &PdfJobs,
    http: reqwest::Client,
    renderer_url: &str,
    exam: &str,
    scanned_pdf: Option<String>,
) -> PdfJob {
    let mut map = jobs.lock().await;
    if let Some(j) = map.get(exam).filter(|j| j.status == PdfJobStatus::Running) {
        return j.clone();
    }
    let job = PdfJob {
        exam: exam.to_string(),
        status: PdfJobStatus::Running,
        scanned_pdf: scanned_pdf.clone(),
        started_at: now_iso(),
        finished_at: None,
        filename: None,
        students: None,
        covers_missing: Vec::new(),
        error: None,
    };
    map.insert(exam.to_string(), job.clone());
    drop(map);

    let url = format!("{}/render", renderer_url.trim_end_matches('/'));
    let jobs = jobs.clone();
    let name = exam.to_string();
    tokio::spawn(async move {
        let outcome = render(&http, &url, &name, scanned_pdf.as_deref()).await;
        if let Some(j) = jobs.lock().await.get_mut(&name) {
            finish(j, outcome);
        }
    });
    job
}

pub async fn get(jobs: &PdfJobs, exam: &str) -> Option<PdfJob> {
    jobs.lock().await.get(exam).cloned()
}

async fn render(
    http: &reqwest::Client,
    url: &str,
    exam: &str,
    scanned_pdf: Option<&str>,
) -> Result<Value, String> {
    let resp = http
        .post(url)
        .timeout(RENDER_TIMEOUT)
        .json(&json!({ "exam": exam, "scanned_pdf": scanned_pdf }))
        .send()
        .await
        .map_err(|e| format!("cannot reach renderer: {e}"))?;
    if !resp.status().is_success() {
        let st = resp.status();
        let t = resp.text().await.unwrap_or_default();
        return Err(format!("renderer {st}: {t}"));
    }
    resp.json().await.map_err(|e| format!("bad renderer response: {e}"))
}

/// Record a finished render. The renderer reports an absolute server `path`;
/// clients only need its file name, served from `graded_exams/`.
fn finish(job: &mut PdfJob, outcome: Result<Value, String>) {
    job.finished_at = Some(now_iso());
    match outcome {
        Ok(v) => {
            job.status = PdfJobStatus::Done;
            job.filename = v
                .get("path")
                .and_then(Value::as_str)
                .and_then(|p| std::path::Path::new(p).file_name())
                .and_then(|n| n.to_str())
                .map(str::to_string);
            job.students = v.get("students").and_then(Value::as_u64).map(|n| n as u32);
            job.covers_missing = v
                .get("covers_missing")
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect())
                .unwrap_or_default();
        }
        Err(e) => {
            job.status = PdfJobStatus::Failed;
            job.error = Some(e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn running() -> PdfJob {
        PdfJob {
            exam: "e".into(),
            status: PdfJobStatus::Running,
            scanned_pdf: None,
            started_at: now_iso(),
            finished_at: None,
            filename: None,
            students: None,
            covers_missing: Vec::new(),
            error: None,
        }
    }

    #[test]
    fn finish_success_keeps_only_file_name() {
        let mut j = running();
        finish(
            &mut j,
            Ok(json!({ "path": "/data/graded_exams/e_results.pdf", "students": 3,
                       "covers_missing": ["42"] })),
        );
        assert_eq!(j.status, PdfJobStatus::Done);
        assert_eq!(j.filename.as_deref(), Some("e_results.pdf"));
        assert_eq!(j.students, Some(3));
        assert_eq!(j.covers_missing, vec!["42"]);
        assert!(j.finished_at.is_some());
    }

    #[test]
    fn finish_failure_records_error() {
        let mut j = running();
        finish(&mut j, Err("renderer 500: boom".into()));
        assert_eq!(j.status, PdfJobStatus::Failed);
        assert_eq!(j.error.as_deref(), Some("renderer 500: boom"));
        assert!(j.filename.is_none());
    }

    #[tokio::test]
    async fn start_reuses_running_job() {
        let jobs: PdfJobs = Arc::default();
        jobs.lock().await.insert("e".into(), running());
        let j = start(&jobs, reqwest::Client::new(), "http://unused", "e", Some("s.pdf".into()))
            .await;
        assert_eq!(j.status, PdfJobStatus::Running);
        assert!(j.scanned_pdf.is_none(), "existing job returned, not a new one");
    }
}
