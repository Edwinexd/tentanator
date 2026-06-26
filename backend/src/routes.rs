//! HTTP routes. DB work goes through the shared mutexed connection
//! (`s.db().await`), held only across DB calls and released across LLM/file I/O.
//! All business logic lives in `store`, `sampling`, `grade`, `llm`.
//!
//! An exam is the central object (`/api/exams/{name}/...`); sessions are
//! lightweight grading passes under an exam. Exam *files* on disk live under
//! `/api/exam-files`.

use std::collections::HashMap;

use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, Path, Query, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post, put};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tower_http::cors::CorsLayer;

use crate::config::Config;
use crate::domain::{
    self, AIGradeSuggestion, Exam, ExamSummary, GradedItem, QuestionGrades, Session, SessionSummary,
};
use crate::error::{AppError, AppResult};
use crate::grade::validate_grade;
use crate::sampling::{self, Algorithm};
use crate::scheme::{self, GradeScheme, StudentResult};
use crate::{llm, store, workspace, AppState};

const NUM_REPRESENTATIVE_SAMPLES: usize = 5;
const DEFAULT_SESSION: &str = "default";
const XLSX_MIME: &str = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet";

/// Build an attachment response so the browser downloads the file.
fn download(filename: &str, content_type: &str, bytes: Vec<u8>) -> Response {
    (
        [
            (header::CONTENT_TYPE, content_type.to_string()),
            (
                header::CONTENT_DISPOSITION,
                format!("attachment; filename=\"{filename}\""),
            ),
        ],
        bytes,
    )
        .into_response()
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/api/health", get(health))
        .route("/api/legacy-workspaces", get(list_legacy_workspaces))
        .route("/api/legacy-workspaces/{name}/import", post(import_workspace))
        .route("/api/legacy-sessions", get(legacy_sessions_info))
        .route("/api/legacy-sessions/import", post(import_legacy_sessions))
        .route("/api/exam-files", get(list_exam_files))
        .route("/api/scans", get(list_scans))
        .route(
            "/api/files/{kind}/{filename}",
            put(upload_file).layer(DefaultBodyLimit::max(512 * 1024 * 1024)),
        )
        .route("/api/exam-files/{file}/columns", get(exam_columns))
        .route("/api/exam-files/{file}/rows", get(exam_rows))
        .route("/api/exams", get(list_exams).post(create_exam))
        .route(
            "/api/exams/{name}",
            get(get_exam).put(update_exam).delete(delete_exam),
        )
        .route("/api/exams/{name}/archive", post(archive_exam))
        .route("/api/exams/{name}/unarchive", post(unarchive_exam))
        .route("/api/exams/{name}/columns", put(update_exam_columns))
        .route(
            "/api/exams/{name}/sessions",
            get(list_sessions).post(create_session),
        )
        .route("/api/exams/{name}/sessions/{session}", delete(delete_session))
        .route("/api/exams/{name}/questions/{col}", put(put_question))
        .route("/api/exams/{name}/questions/{col}/sampling", post(run_sampling))
        .route("/api/exams/{name}/questions/{col}/grade", post(grade_item))
        .route(
            "/api/exams/{name}/questions/{col}/grade/{row_id}",
            delete(ungrade_item),
        )
        .route("/api/exams/{name}/questions/{col}/suggest", post(suggest))
        .route("/api/exams/{name}/questions/{col}/status", get(question_status))
        .route("/api/exams/{name}/scheme", put(put_scheme))
        .route("/api/exams/{name}/questions-config", put(put_questions_config))
        .route("/api/exams/{name}/results", get(get_results).post(preview_results))
        .route("/api/exams/{name}/import/preview", post(import_preview))
        .route("/api/exams/{name}/import/apply", post(import_apply))
        .route("/api/exams/{name}/conflicts", get(get_conflicts))
        .route("/api/exams/{name}/conflicts/resolve", post(resolve_conflict))
        .route("/api/exams/{name}/export", post(export))
        .route("/api/exams/{name}/export/daisy", post(export_daisy_route))
        .route("/api/exams/{name}/export/csv", post(export_csv_route))
        .route("/api/exams/{name}/render-data", get(render_data))
        .route("/api/exams/{name}/export/results-pdf", post(export_results_pdf))
        .route("/api/graded/{filename}", get(download_graded))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

// ---------------------------------------------------------------------------
// Request DTOs
// ---------------------------------------------------------------------------

#[derive(Deserialize, Default)]
struct ListQuery {
    #[serde(default)]
    archived: bool,
    #[serde(default)]
    course: Option<String>,
}

#[derive(Deserialize)]
struct CreateExam {
    exam_file: String,
    id_columns: Vec<String>,
    input_columns: Vec<String>,
    output_columns: Vec<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    course: Option<String>,
}

#[derive(Deserialize)]
struct ExamMeta {
    course: Option<String>,
}

#[derive(Deserialize)]
struct CreateSessionReq {
    #[serde(default)]
    name: Option<String>,
}

#[derive(Deserialize)]
struct QuestionMeta {
    exam_question: Option<String>,
    sample_answer: Option<String>,
    global_question_id: Option<String>,
}

#[derive(Deserialize)]
struct SamplingReq {
    algorithm: Algorithm,
    n_samples: Option<usize>,
}

#[derive(Deserialize)]
struct GradeReq {
    row_id: String,
    grade: String,
    #[serde(default)]
    session: Option<String>,
}

#[derive(Deserialize)]
struct SuggestReq {
    row_id: String,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn exam_data(
    config: &Config,
    exam: &Exam,
) -> AppResult<(Vec<HashMap<String, String>>, Vec<String>)> {
    let path = store::resolve_exam_path(config, &exam.exam_file).ok_or_else(|| {
        AppError::NotFound(format!("exam file '{}' not found", exam.exam_file))
    })?;
    let rows = store::read_exam_data(&path)?;
    let cols = store::get_exam_columns(&path)?;
    Ok((rows, cols))
}

async fn load_exam_or_404(s: &AppState, name: &str) -> AppResult<Exam> {
    let conn = s.db().await;
    store::load_exam(&conn, name)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("exam '{name}' not found")))
}

fn no_question(col: &str) -> AppError {
    AppError::NotFound(format!("question '{col}' not found"))
}

// ---------------------------------------------------------------------------
// Health & exam files (on disk)
// ---------------------------------------------------------------------------

async fn health() -> Json<Value> {
    Json(json!({ "status": "ok" }))
}

// ---------------------------------------------------------------------------
// Legacy import (old Python-app data -> new format, on demand). Not run on
// startup; the user triggers it from the home screen.
// ---------------------------------------------------------------------------

async fn list_legacy_workspaces(
    State(s): State<AppState>,
) -> Json<Vec<workspace::WorkspaceInfo>> {
    Json(workspace::list_importable(&s.config))
}

async fn import_workspace(
    State(s): State<AppState>,
    Path(name): Path<String>,
) -> AppResult<Json<workspace::ImportResult>> {
    let conn = s.db().await;
    Ok(Json(workspace::import_workspace(&conn, &s.config, &name).await?))
}

/// How many loose legacy sessions sit at the data root (`.tentanator_sessions/`).
async fn legacy_sessions_info(State(s): State<AppState>) -> Json<Value> {
    let active = store::count_session_files(&s.config.sessions_dir());
    let archived = store::count_session_files(&s.config.archive_dir());
    Json(json!({ "count": active + archived }))
}

/// Import the loose root-level `.tentanator_sessions/` (active + archive) plus
/// the root graded pool into the new exam-centric store.
async fn import_legacy_sessions(State(s): State<AppState>) -> AppResult<Json<Value>> {
    let conn = s.db().await;
    let mut imported = store::import_session_dir(&conn, &s.config.sessions_dir(), false).await?;
    imported.extend(store::import_session_dir(&conn, &s.config.archive_dir(), true).await?);
    store::import_pool_dir(&conn, &s.config.graded_pool_dir()).await?;
    Ok(Json(json!({ "imported_exams": imported })))
}

async fn list_exam_files(State(s): State<AppState>) -> Json<Vec<String>> {
    Json(store::list_exam_files(&s.config))
}

async fn list_scans(State(s): State<AppState>) -> Json<Vec<String>> {
    Json(store::list_pdf_files(&s.config))
}

/// Upload a file (raw body) into exams/ or scans/. The filename is reduced to a
/// basename (no path traversal). Used by the web/TUI file pickers.
async fn upload_file(
    State(s): State<AppState>,
    Path((kind, filename)): Path<(String, String)>,
    body: Bytes,
) -> AppResult<Json<Value>> {
    let dir = match kind.as_str() {
        "exams" => s.config.exams_dir(),
        "scans" => s.config.data_dir.join("scans"),
        _ => return Err(AppError::BadRequest("kind must be 'exams' or 'scans'".into())),
    };
    let fname = std::path::Path::new(&filename)
        .file_name()
        .and_then(|n| n.to_str())
        .filter(|n| !n.is_empty())
        .ok_or_else(|| AppError::BadRequest("invalid filename".into()))?;
    std::fs::create_dir_all(&dir)?;
    std::fs::write(dir.join(fname), &body)?;
    Ok(Json(json!({ "filename": fname })))
}

async fn exam_columns(
    State(s): State<AppState>,
    Path(file): Path<String>,
) -> AppResult<Json<Vec<String>>> {
    let path = store::resolve_exam_path(&s.config, &file)
        .ok_or_else(|| AppError::NotFound(format!("exam file '{file}' not found")))?;
    Ok(Json(store::get_exam_columns(&path)?))
}

async fn exam_rows(
    State(s): State<AppState>,
    Path(file): Path<String>,
) -> AppResult<Json<Value>> {
    let path = store::resolve_exam_path(&s.config, &file)
        .ok_or_else(|| AppError::NotFound(format!("exam file '{file}' not found")))?;
    let rows = store::read_exam_data(&path)?;
    Ok(Json(json!({ "rows": rows })))
}

// ---------------------------------------------------------------------------
// Exams
// ---------------------------------------------------------------------------

async fn list_exams(
    State(s): State<AppState>,
    Query(q): Query<ListQuery>,
) -> AppResult<Json<Vec<ExamSummary>>> {
    let mut exams = {
        let conn = s.db().await;
        store::list_exams(&conn, q.archived).await?
    };
    if let Some(course) = q.course.filter(|c| !c.is_empty()) {
        exams.retain(|x| x.course.as_deref() == Some(course.as_str()));
    }
    Ok(Json(exams))
}

async fn create_exam(
    State(s): State<AppState>,
    Json(req): Json<CreateExam>,
) -> AppResult<Json<Exam>> {
    if req.input_columns.is_empty() || req.output_columns.is_empty() {
        return Err(AppError::BadRequest(
            "input_columns and output_columns are required".into(),
        ));
    }
    let raw_name = req.name.clone().unwrap_or_else(|| {
        let stem = std::path::Path::new(&req.exam_file)
            .file_stem()
            .and_then(|x| x.to_str())
            .unwrap_or("exam");
        format!("{stem}_{}", store::timestamp_compact())
    });

    let mut exam = Exam {
        name: store::sanitize_name(&raw_name),
        exam_file: req.exam_file,
        id_columns: if req.id_columns.is_empty() {
            vec!["_row_number".to_string()]
        } else {
            req.id_columns
        },
        input_columns: req.input_columns,
        output_columns: req.output_columns.clone(),
        course: req.course.filter(|c| !c.is_empty()),
        last_updated: store::now_iso(),
        questions: HashMap::new(),
        scheme: None,
    };
    for col in &req.output_columns {
        exam.ensure_question(col);
    }
    // Seed grades already present in the sheet (auto-scored questions) so they
    // are not re-graded by hand. Read the file before taking the DB lock.
    let prefilled = exam_data(&s.config, &exam).map(|(rows, _)| rows).unwrap_or_default();
    {
        let conn = s.db().await;
        store::insert_exam(&conn, &exam).await?;
        // Every exam starts with a default grading session.
        store::create_session(&conn, &exam.name, DEFAULT_SESSION).await?;
        store::seed_prefilled_grades(&conn, &exam, &prefilled, &exam.output_columns).await?;
    }
    let conn = s.db().await;
    Ok(Json(store::load_exam(&conn, &exam.name).await?.unwrap_or(exam)))
}

async fn get_exam(
    State(s): State<AppState>,
    Path(name): Path<String>,
) -> AppResult<Json<Exam>> {
    Ok(Json(load_exam_or_404(&s, &name).await?))
}

async fn update_exam(
    State(s): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<ExamMeta>,
) -> AppResult<Json<Exam>> {
    let conn = s.db().await;
    if !store::exam_exists(&conn, &name).await? {
        return Err(AppError::NotFound(format!("exam '{name}' not found")));
    }
    if let Some(course) = req.course {
        let course = if course.is_empty() { None } else { Some(course) };
        store::set_course(&conn, &name, course.as_deref()).await?;
    }
    store::touch(&conn, &name).await?;
    store::load_exam(&conn, &name)
        .await?
        .map(Json)
        .ok_or_else(|| AppError::NotFound(format!("exam '{name}' not found")))
}

async fn delete_exam(
    State(s): State<AppState>,
    Path(name): Path<String>,
) -> AppResult<StatusCode> {
    let conn = s.db().await;
    if store::delete_exam(&conn, &name).await? {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(AppError::NotFound(format!("exam '{name}' not found")))
    }
}

async fn archive_exam(
    State(s): State<AppState>,
    Path(name): Path<String>,
) -> AppResult<StatusCode> {
    let conn = s.db().await;
    if store::set_archived(&conn, &name, true).await? {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(AppError::NotFound(format!("exam '{name}' not found")))
    }
}

async fn unarchive_exam(
    State(s): State<AppState>,
    Path(name): Path<String>,
) -> AppResult<StatusCode> {
    let conn = s.db().await;
    if store::set_archived(&conn, &name, false).await? {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(AppError::NotFound(format!("exam '{name}' not found")))
    }
}

#[derive(Deserialize)]
struct ColumnsReq {
    #[serde(default)]
    id_columns: Vec<String>,
    input_columns: Vec<String>,
    output_columns: Vec<String>,
}

/// Replace an exam's column mapping. Adds a question for each new output column
/// (index-paired with its input); existing questions and their grades are kept.
/// Used to expand a partially-configured exam to all its question columns.
async fn update_exam_columns(
    State(s): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<ColumnsReq>,
) -> AppResult<Json<Exam>> {
    if req.input_columns.is_empty() || req.output_columns.is_empty() {
        return Err(AppError::BadRequest(
            "input_columns and output_columns are required".into(),
        ));
    }
    let mut exam = load_exam_or_404(&s, &name).await?;
    // Columns that already had a question keep their grades; only newly-added
    // ones get seeded from the sheet below.
    let existing: std::collections::HashSet<String> = exam.questions.keys().cloned().collect();
    exam.id_columns = if req.id_columns.is_empty() {
        vec!["_row_number".to_string()]
    } else {
        req.id_columns
    };
    exam.input_columns = req.input_columns;
    exam.output_columns = req.output_columns.clone();
    for col in &req.output_columns {
        exam.ensure_question(col);
    }
    let new_cols: Vec<String> = req
        .output_columns
        .iter()
        .filter(|c| !existing.contains(*c))
        .cloned()
        .collect();
    let prefilled = exam_data(&s.config, &exam).map(|(rows, _)| rows).unwrap_or_default();
    {
        let conn = s.db().await;
        store::set_columns(
            &conn,
            &name,
            &exam.id_columns,
            &exam.input_columns,
            &exam.output_columns,
        )
        .await?;
        for col in &req.output_columns {
            if let Some(q) = exam.questions.get(col) {
                store::upsert_question_row(&conn, &name, col, q).await?;
            }
        }
        store::seed_prefilled_grades(&conn, &exam, &prefilled, &new_cols).await?;
        store::touch(&conn, &name).await?;
    }
    Ok(Json(load_exam_or_404(&s, &name).await?))
}

// ---------------------------------------------------------------------------
// Sessions (lightweight grading passes under an exam)
// ---------------------------------------------------------------------------

async fn list_sessions(
    State(s): State<AppState>,
    Path(name): Path<String>,
) -> AppResult<Json<Vec<SessionSummary>>> {
    let conn = s.db().await;
    if !store::exam_exists(&conn, &name).await? {
        return Err(AppError::NotFound(format!("exam '{name}' not found")));
    }
    Ok(Json(store::list_sessions(&conn, &name).await?))
}

async fn create_session(
    State(s): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<CreateSessionReq>,
) -> AppResult<Json<Session>> {
    let conn = s.db().await;
    if !store::exam_exists(&conn, &name).await? {
        return Err(AppError::NotFound(format!("exam '{name}' not found")));
    }
    let raw = req
        .name
        .filter(|n| !n.trim().is_empty())
        .unwrap_or_else(|| format!("session_{}", store::timestamp_compact()));
    let session_name = store::sanitize_name(&raw);
    Ok(Json(store::create_session(&conn, &name, &session_name).await?))
}

async fn delete_session(
    State(s): State<AppState>,
    Path((name, session)): Path<(String, String)>,
) -> AppResult<StatusCode> {
    if session == DEFAULT_SESSION {
        return Err(AppError::BadRequest(
            "the default session cannot be deleted".into(),
        ));
    }
    let conn = s.db().await;
    if store::delete_session(&conn, &name, &session).await? {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(AppError::NotFound(format!("session '{session}' not found")))
    }
}

// ---------------------------------------------------------------------------
// Questions & grading
// ---------------------------------------------------------------------------

async fn put_question(
    State(s): State<AppState>,
    Path((name, col)): Path<(String, String)>,
    Json(req): Json<QuestionMeta>,
) -> AppResult<Json<QuestionGrades>> {
    let mut exam = load_exam_or_404(&s, &name).await?;
    {
        let q = exam.ensure_question(&col);
        if let Some(x) = req.exam_question {
            q.exam_question = x;
        }
        if let Some(x) = req.sample_answer {
            q.sample_answer = x;
        }
        if let Some(x) = req.global_question_id {
            q.global_question_id = if x.is_empty() { None } else { Some(x) };
        }
    }
    let q = exam.questions.get(&col).cloned().ok_or_else(|| no_question(&col))?;
    let conn = s.db().await;
    store::upsert_question_row(&conn, &name, &col, &q).await?;
    // Linking to a global question id makes prior grades sharable as ICL examples.
    store::sync_question_grades_to_pool(&conn, &name, &col).await?;
    store::touch(&conn, &name).await?;
    store::load_question(&conn, &name, &col)
        .await?
        .map(Json)
        .ok_or_else(|| no_question(&col))
}

async fn run_sampling(
    State(s): State<AppState>,
    Path((name, col)): Path<(String, String)>,
    Json(req): Json<SamplingReq>,
) -> AppResult<Json<crate::domain::SamplingResult>> {
    let mut exam = load_exam_or_404(&s, &name).await?;
    exam.ensure_question(&col);
    let input_column = exam.questions[&col].input_column.clone();
    let id_columns = exam.id_columns.clone();
    let (rows, _) = exam_data(&s.config, &exam)?;

    let candidates: Vec<(String, String)> = rows
        .iter()
        .filter_map(|row| {
            let rid = domain::row_id(row, &id_columns);
            let text = row.get(&input_column).cloned().unwrap_or_default();
            domain::is_meaningful(&text).then_some((rid, text))
        })
        .collect();
    let n = req.n_samples.unwrap_or(NUM_REPRESENTATIVE_SAMPLES);

    let selected = match req.algorithm {
        Algorithm::Random => {
            let ids: Vec<String> = candidates.iter().map(|(rid, _)| rid.clone()).collect();
            sampling::random_sample(&ids, n)
        }
        Algorithm::Maximin => {
            let existing: std::collections::HashSet<String> = {
                let conn = s.db().await;
                store::load_feature_cache(&conn, &name, &input_column).await?
            }
            .into_keys()
            .collect();
            let to_embed: Vec<(String, String)> = candidates
                .iter()
                .filter(|(rid, _)| !existing.contains(rid))
                .cloned()
                .collect();
            if !to_embed.is_empty() {
                let texts: Vec<String> = to_embed.iter().map(|(_, t)| t.clone()).collect();
                let embs = llm::embed_many(&s.config, &s.http, &texts).await;
                let conn = s.db().await;
                for ((rid, _), emb) in to_embed.iter().zip(embs) {
                    if let Some(v) = emb {
                        store::put_feature_vector(&conn, &name, &input_column, rid, &v).await?;
                    }
                }
            }
            let cache = {
                let conn = s.db().await;
                store::load_feature_cache(&conn, &name, &input_column).await?
            };
            let valid: Vec<(String, Vec<f32>)> = candidates
                .iter()
                .filter_map(|(rid, _)| {
                    cache
                        .get(rid)
                        .filter(|v| !v.is_empty())
                        .map(|v| (rid.clone(), v.clone()))
                })
                .collect();
            sampling::maximin_sample(&valid, n)
        }
    };

    let result = crate::domain::SamplingResult {
        algorithm: req.algorithm.as_str().to_string(),
        selected_ids: selected.clone(),
        quality_score: 0.0,
        num_samples: selected.len(),
        timestamp: store::now_iso(),
    };
    {
        let q = exam.ensure_question(&col);
        q.sampling_result = Some(result.clone());
    }
    let q = exam.questions.get(&col).cloned().ok_or_else(|| no_question(&col))?;
    let conn = s.db().await;
    store::upsert_question_row(&conn, &name, &col, &q).await?;
    store::touch(&conn, &name).await?;
    Ok(Json(result))
}

async fn grade_item(
    State(s): State<AppState>,
    Path((name, col)): Path<(String, String)>,
    Json(req): Json<GradeReq>,
) -> AppResult<Json<QuestionGrades>> {
    validate_grade(&req.grade).map_err(AppError::BadRequest)?;
    let mut exam = load_exam_or_404(&s, &name).await?;
    exam.ensure_question(&col);
    let input_column = exam.questions[&col].input_column.clone();
    let id_columns = exam.id_columns.clone();
    let session = req
        .session
        .as_deref()
        .filter(|x| !x.trim().is_empty())
        .unwrap_or(DEFAULT_SESSION)
        .to_string();

    let (rows, _) = exam_data(&s.config, &exam)?;
    let input_text = rows
        .iter()
        .find(|r| domain::row_id(r, &id_columns) == req.row_id)
        .and_then(|r| r.get(&input_column).cloned())
        .unwrap_or_default();

    // Persist the question row first so the pool sync can read its global id.
    let q = exam.questions.get(&col).cloned().ok_or_else(|| no_question(&col))?;
    {
        let conn = s.db().await;
        store::upsert_question_row(&conn, &name, &col, &q).await?;
        store::create_session(&conn, &name, &session).await?;
    }

    // Cache a feature vector so the row can participate in future sampling.
    if domain::is_meaningful(&input_text) {
        let cached = {
            let conn = s.db().await;
            store::has_feature_vector(&conn, &name, &input_column, &req.row_id).await?
        };
        if !cached {
            if let Ok(v) = llm::embed(&s.config, &s.http, &input_text).await {
                let conn = s.db().await;
                store::put_feature_vector(&conn, &name, &input_column, &req.row_id, &v).await?;
            }
        }
    }

    let item = GradedItem {
        row_id: req.row_id.clone(),
        input_text,
        grade: req.grade.clone(),
        timestamp: store::now_iso(),
    };
    let conn = s.db().await;
    store::put_graded_item(&conn, &name, &col, &item, "manual", &session).await?;
    store::touch_session(&conn, &name, &session).await?;
    store::touch(&conn, &name).await?;
    store::load_question(&conn, &name, &col)
        .await?
        .map(Json)
        .ok_or_else(|| no_question(&col))
}

async fn ungrade_item(
    State(s): State<AppState>,
    Path((name, col, row_id)): Path<(String, String, String)>,
) -> AppResult<Json<QuestionGrades>> {
    let conn = s.db().await;
    store::delete_graded_item(&conn, &name, &col, &row_id).await?;
    store::touch(&conn, &name).await?;
    store::load_question(&conn, &name, &col)
        .await?
        .map(Json)
        .ok_or_else(|| no_question(&col))
}

async fn suggest(
    State(s): State<AppState>,
    Path((name, col)): Path<(String, String)>,
    Json(req): Json<SuggestReq>,
) -> AppResult<Json<AIGradeSuggestion>> {
    let exam = load_exam_or_404(&s, &name).await?;
    let question = exam.questions.get(&col).ok_or_else(|| no_question(&col))?;
    if !llm::has_enough_icl(question) {
        return Err(AppError::BadRequest(format!(
            "AI suggestions need at least {} graded examples for this question",
            llm::MIN_ICL_EXAMPLES
        )));
    }

    let id_columns = exam.id_columns.clone();
    let input_column = question.input_column.clone();
    let (rows, _) = exam_data(&s.config, &exam)?;
    let response_text = rows
        .iter()
        .find(|r| domain::row_id(r, &id_columns) == req.row_id)
        .and_then(|r| r.get(&input_column).cloned())
        .ok_or_else(|| AppError::NotFound(format!("row '{}' not found", req.row_id)))?;

    let suggestion = llm::suggest_grade(&s.config, &s.http, question, &response_text).await?;
    Ok(Json(suggestion))
}

async fn question_status(
    State(s): State<AppState>,
    Path((name, col)): Path<(String, String)>,
) -> AppResult<Json<Value>> {
    let exam = load_exam_or_404(&s, &name).await?;
    let q = exam.questions.get(&col).ok_or_else(|| no_question(&col))?;
    let external = q
        .external_graded_items
        .iter()
        .filter(|i| domain::is_meaningful(&i.input_text))
        .count();
    Ok(Json(json!({
        "graded": q.graded_items.len(),
        "valid_graded": q.valid_graded_count(),
        "external": external,
        "icl_ready": llm::has_enough_icl(q),
        "min_icl_examples": llm::MIN_ICL_EXAMPLES,
        "sampling_result": q.sampling_result,
    })))
}

async fn export(
    State(s): State<AppState>,
    Path(name): Path<String>,
) -> AppResult<Response> {
    let exam = load_exam_or_404(&s, &name).await?;
    let (rows, cols) = exam_data(&s.config, &exam)?;
    let (filename, bytes) = store::build_graded_xlsx(&exam, &rows, &cols)?;
    Ok(download(&filename, XLSX_MIME, bytes))
}

// ---------------------------------------------------------------------------
// Examination scheme, question config & computed results
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct QuestionConfigUpdate {
    col: String,
    #[serde(default)]
    var: String,
    #[serde(default)]
    group: String,
    #[serde(default)]
    qtype: String,
    #[serde(default)]
    max_points: f64,
    #[serde(default)]
    position: i32,
    #[serde(default)]
    estimate: Option<String>,
}

#[derive(Serialize)]
struct ResultsResponse {
    results: Vec<StudentResult>,
    distribution: HashMap<String, usize>,
    total_students: usize,
    complete: usize,
    has_scheme: bool,
    unresolved_conflicts: usize,
}

fn sanitize_ident(s: &str) -> String {
    let mut out: String = s
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' })
        .collect();
    if out.is_empty() || out.chars().next().unwrap().is_ascii_digit() {
        out.insert(0, '_');
    }
    out
}

/// Variable name a question contributes to scheme expressions.
fn effective_var(col: &str, q: &QuestionGrades) -> String {
    if q.var.is_empty() {
        sanitize_ident(col)
    } else {
        q.var.clone()
    }
}

fn compute_results(
    exam: &Exam,
    rows: &[HashMap<String, String>],
    scheme: &GradeScheme,
) -> ResultsResponse {
    let questions: Vec<scheme::QuestionConfig> = exam
        .output_columns
        .iter()
        .filter_map(|col| exam.questions.get(col).map(|q| (col, q)))
        .map(|(col, q)| scheme::QuestionConfig {
            var: effective_var(col, q),
            label: q.question_name.clone(),
            group: q.group.clone(),
            qtype: q.qtype.clone(),
            max_points: q.max_points,
            position: q.position,
            estimate: q.estimate.clone(),
        })
        .collect();

    let mut results = Vec::with_capacity(rows.len());
    for row in rows {
        let rid = domain::row_id(row, &exam.id_columns);
        let mut points: HashMap<String, Option<f64>> = HashMap::new();
        for col in &exam.output_columns {
            if let Some(q) = exam.questions.get(col) {
                let var = effective_var(col, q);
                let graded = q
                    .graded_items
                    .iter()
                    .find(|gi| gi.row_id == rid)
                    .and_then(|gi| crate::grade::evaluate_grade(&gi.grade));
                points.insert(var, graded);
            }
        }
        results.push(scheme::compute_student(&rid, scheme, &questions, &points));
    }

    let distribution = scheme::distribution(&results);
    let complete = results.iter().filter(|r| r.complete).count();
    let total_students = results.len();
    ResultsResponse {
        results,
        distribution,
        total_students,
        complete,
        has_scheme: true,
        unresolved_conflicts: 0,
    }
}

/// Effective scheme variables for an exam's questions (in column order).
fn exam_question_vars(exam: &Exam) -> Vec<String> {
    exam.output_columns
        .iter()
        .filter_map(|c| exam.questions.get(c).map(|q| effective_var(c, q)))
        .collect()
}

async fn put_scheme(
    State(s): State<AppState>,
    Path(name): Path<String>,
    Json(scheme): Json<GradeScheme>,
) -> AppResult<StatusCode> {
    let exam = load_exam_or_404(&s, &name).await?;
    scheme::validate_scheme(&scheme, &exam_question_vars(&exam))
        .map_err(AppError::BadRequest)?;
    let conn = s.db().await;
    store::set_scheme(&conn, &name, &Some(scheme)).await?;
    store::touch(&conn, &name).await?;
    Ok(StatusCode::NO_CONTENT)
}

async fn put_questions_config(
    State(s): State<AppState>,
    Path(name): Path<String>,
    Json(updates): Json<Vec<QuestionConfigUpdate>>,
) -> AppResult<Json<Exam>> {
    let mut exam = load_exam_or_404(&s, &name).await?;
    for u in &updates {
        let q = exam.ensure_question(&u.col);
        q.var = u.var.clone();
        q.group = u.group.clone();
        q.qtype = u.qtype.clone();
        q.max_points = u.max_points;
        q.position = u.position;
        q.estimate = u.estimate.clone().filter(|e| !e.is_empty());
    }
    // Each question must contribute a unique scheme variable, else points from
    // colliding questions would silently merge during compute.
    let mut seen = std::collections::HashSet::new();
    for col in &exam.output_columns {
        if let Some(q) = exam.questions.get(col) {
            let v = effective_var(col, q);
            if !seen.insert(v.clone()) {
                return Err(AppError::BadRequest(format!(
                    "duplicate scheme variable `{v}` - each question needs a unique var"
                )));
            }
        }
    }
    {
        let conn = s.db().await;
        for u in &updates {
            let qc = exam
                .questions
                .get(&u.col)
                .cloned()
                .ok_or_else(|| no_question(&u.col))?;
            store::upsert_question_row(&conn, &name, &u.col, &qc).await?;
        }
        store::touch(&conn, &name).await?;
    }
    Ok(Json(load_exam_or_404(&s, &name).await?))
}

async fn get_results(
    State(s): State<AppState>,
    Path(name): Path<String>,
) -> AppResult<Json<ResultsResponse>> {
    let exam = load_exam_or_404(&s, &name).await?;
    let (rows, _) = exam_data(&s.config, &exam)?;
    // Seed any grades from prefilled spreadsheet values not yet in the DB.
    // `put_graded_item` is an upsert, so this is safe to run every time.
    {
        let conn = s.db().await;
        store::seed_prefilled_grades(&conn, &exam, &rows, &exam.output_columns).await?;
    }
    // Reload to pick up newly seeded grades.
    let exam = load_exam_or_404(&s, &name).await?;
    let mut resp = match &exam.scheme {
        Some(scheme) => compute_results(&exam, &rows, scheme),
        None => ResultsResponse {
            results: Vec::new(),
            distribution: HashMap::new(),
            total_students: rows.len(),
            complete: 0,
            has_scheme: false,
            unresolved_conflicts: 0,
        },
    };
    let conn = s.db().await;
    resp.unresolved_conflicts = store::count_conflicts(&conn, &name).await?;
    Ok(Json(resp))
}

async fn preview_results(
    State(s): State<AppState>,
    Path(name): Path<String>,
    Json(scheme): Json<GradeScheme>,
) -> AppResult<Json<ResultsResponse>> {
    let exam = load_exam_or_404(&s, &name).await?;
    scheme::validate_scheme(&scheme, &exam_question_vars(&exam))
        .map_err(AppError::BadRequest)?;
    let (rows, _) = exam_data(&s.config, &exam)?;
    Ok(Json(compute_results(&exam, &rows, &scheme)))
}

// ---------------------------------------------------------------------------
// Import & merge graded sheets
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct ColMapping {
    column: String,
    output_col: String,
}

#[derive(Deserialize)]
struct ImportReq {
    file: String,
    id_column: String,
    mappings: Vec<ColMapping>,
    #[serde(default)]
    label: Option<String>,
}

#[derive(Deserialize)]
struct ResolveReq {
    output_col: String,
    row_id: String,
    choose: String, // "existing" | "incoming"
}

#[derive(Serialize)]
struct ConflictSample {
    output_col: String,
    row_id: String,
    existing: String,
    incoming: String,
}

#[derive(Default, Serialize)]
struct ImportSummary {
    new: usize,
    same: usize,
    conflict: usize,
    skipped: usize,
    unknown_ids: usize,
    conflicts: Vec<ConflictSample>,
}

fn grades_equal(a: &str, b: &str) -> bool {
    match (crate::grade::evaluate_grade(a), crate::grade::evaluate_grade(b)) {
        (Some(x), Some(y)) => (x - y).abs() < 1e-9,
        _ => a.trim().eq_ignore_ascii_case(b.trim()),
    }
}

/// Read roster + import file, returning (roster id-set, import rows).
fn load_import(
    s: &AppState,
    exam: &Exam,
    file: &str,
) -> AppResult<(std::collections::HashSet<String>, Vec<HashMap<String, String>>)> {
    let (roster, _) = exam_data(&s.config, exam)?;
    let ids: std::collections::HashSet<String> = roster
        .iter()
        .map(|r| domain::row_id(r, &exam.id_columns))
        .collect();
    let ipath = store::resolve_exam_path(&s.config, file)
        .ok_or_else(|| AppError::NotFound(format!("import file '{file}' not found")))?;
    let irows = store::read_exam_data(&ipath)?;
    Ok((ids, irows))
}

async fn import_preview(
    State(s): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<ImportReq>,
) -> AppResult<Json<ImportSummary>> {
    let conn = s.db().await;
    let exam = store::load_exam(&conn, &name)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("exam '{name}' not found")))?;
    let (roster_ids, irows) = load_import(&s, &exam, &req.file)?;

    let mut summary = ImportSummary::default();
    for irow in &irows {
        let id = irow.get(&req.id_column).cloned().unwrap_or_default();
        if id.is_empty() {
            continue;
        }
        if !roster_ids.contains(&id) {
            summary.unknown_ids += 1;
            continue;
        }
        for m in &req.mappings {
            let incoming = irow.get(&m.column).cloned().unwrap_or_default();
            if !domain::is_meaningful(&incoming) {
                summary.skipped += 1;
                continue;
            }
            match store::get_graded(&conn, &name, &m.output_col, &id).await? {
                None => summary.new += 1,
                Some((existing, _)) => {
                    if grades_equal(&existing, &incoming) {
                        summary.same += 1;
                    } else {
                        summary.conflict += 1;
                        if summary.conflicts.len() < 100 {
                            summary.conflicts.push(ConflictSample {
                                output_col: m.output_col.clone(),
                                row_id: id.clone(),
                                existing,
                                incoming,
                            });
                        }
                    }
                }
            }
        }
    }
    Ok(Json(summary))
}

async fn import_apply(
    State(s): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<ImportReq>,
) -> AppResult<Json<ImportSummary>> {
    let conn = s.db().await;
    let exam = store::load_exam(&conn, &name)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("exam '{name}' not found")))?;
    let (roster, _) = exam_data(&s.config, &exam)?;
    let roster_by_id: HashMap<String, &HashMap<String, String>> = roster
        .iter()
        .map(|r| (domain::row_id(r, &exam.id_columns), r))
        .collect();
    let (roster_ids, irows) = load_import(&s, &exam, &req.file)?;
    let label = format!(
        "imported:{}",
        req.label.clone().unwrap_or_else(|| req.file.clone())
    );

    let mut summary = ImportSummary::default();
    for irow in &irows {
        let id = irow.get(&req.id_column).cloned().unwrap_or_default();
        if id.is_empty() {
            continue;
        }
        if !roster_ids.contains(&id) {
            summary.unknown_ids += 1;
            continue;
        }
        for m in &req.mappings {
            let incoming = irow.get(&m.column).cloned().unwrap_or_default();
            if !domain::is_meaningful(&incoming) {
                summary.skipped += 1;
                continue;
            }
            let input_text = exam
                .questions
                .get(&m.output_col)
                .and_then(|q| roster_by_id.get(&id).and_then(|r| r.get(&q.input_column)))
                .cloned()
                .unwrap_or_default();
            match store::get_graded(&conn, &name, &m.output_col, &id).await? {
                Some((existing, existing_source)) if !grades_equal(&existing, &incoming) => {
                    summary.conflict += 1;
                    store::add_conflict(
                        &conn,
                        &name,
                        &domain::GradeConflict {
                            output_col: m.output_col.clone(),
                            row_id: id.clone(),
                            existing_grade: existing,
                            existing_source,
                            incoming_grade: incoming.clone(),
                            incoming_source: label.clone(),
                            input_text,
                            timestamp: store::now_iso(),
                        },
                    )
                    .await?;
                }
                other => {
                    if other.is_some() {
                        // Identical grade already present: leave it (and its
                        // source provenance) untouched.
                        summary.same += 1;
                    } else {
                        summary.new += 1;
                        let item = GradedItem {
                            row_id: id.clone(),
                            input_text,
                            grade: incoming.clone(),
                            timestamp: store::now_iso(),
                        };
                        store::put_graded_item(&conn, &name, &m.output_col, &item, &label, "").await?;
                    }
                }
            }
        }
    }
    store::touch(&conn, &name).await?;
    Ok(Json(summary))
}

async fn export_daisy_route(
    State(s): State<AppState>,
    Path(name): Path<String>,
) -> AppResult<Response> {
    let exam = load_exam_or_404(&s, &name).await?;
    let (rows, _) = exam_data(&s.config, &exam)?;
    let scheme = exam
        .scheme
        .as_ref()
        .ok_or_else(|| AppError::BadRequest("no grade scheme configured".into()))?;
    let resp = compute_results(&exam, &rows, scheme);
    let (filename, bytes) = store::build_daisy_xlsx(&exam, &resp.results)?;
    Ok(download(&filename, XLSX_MIME, bytes))
}

async fn export_csv_route(
    State(s): State<AppState>,
    Path(name): Path<String>,
) -> AppResult<Response> {
    let exam = load_exam_or_404(&s, &name).await?;
    let (rows, _) = exam_data(&s.config, &exam)?;
    let results = exam
        .scheme
        .as_ref()
        .map(|sc| compute_results(&exam, &rows, sc).results)
        .unwrap_or_default();
    let (filename, bytes) = store::build_per_question_csv(&exam, &rows, &results)?;
    Ok(download(&filename, "text/csv; charset=utf-8", bytes))
}

/// Stream a previously-produced file from `graded_exams/` (e.g. the results PDF).
async fn download_graded(
    State(s): State<AppState>,
    Path(filename): Path<String>,
) -> AppResult<Response> {
    let fname = std::path::Path::new(&filename)
        .file_name()
        .and_then(|n| n.to_str())
        .filter(|n| !n.is_empty())
        .ok_or_else(|| AppError::BadRequest("invalid filename".into()))?;
    let path = s.config.graded_dir().join(fname);
    let bytes = std::fs::read(&path)
        .map_err(|_| AppError::NotFound(format!("file '{fname}' not found")))?;
    let ct = if fname.ends_with(".pdf") {
        "application/pdf"
    } else if fname.ends_with(".csv") {
        "text/csv; charset=utf-8"
    } else if fname.ends_with(".xlsx") {
        XLSX_MIME
    } else {
        "application/octet-stream"
    };
    Ok(download(fname, ct, bytes))
}

// ---------------------------------------------------------------------------
// Results-PDF render data + trigger
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct RenderQuestion {
    label: String,
    group: String,
    qtype: String,
    response: String,
    points: Option<f64>,
    max: f64,
    estimated: bool,
}
#[derive(Serialize)]
struct RenderStudent {
    id: String,
    grade: String,
    total: f64,
    questions: Vec<RenderQuestion>,
}
#[derive(Serialize)]
struct RenderData {
    exam: String,
    students: Vec<RenderStudent>,
}
#[derive(Deserialize)]
struct ResultsPdfReq {
    #[serde(default)]
    scanned_pdf: Option<String>,
}

async fn render_data(
    State(s): State<AppState>,
    Path(name): Path<String>,
) -> AppResult<Json<RenderData>> {
    let exam = load_exam_or_404(&s, &name).await?;
    let (rows, _) = exam_data(&s.config, &exam)?;
    let scheme = exam.scheme.clone().unwrap_or_default();
    let questions: Vec<scheme::QuestionConfig> = exam
        .output_columns
        .iter()
        .filter_map(|col| exam.questions.get(col).map(|q| (col, q)))
        .map(|(col, q)| scheme::QuestionConfig {
            var: effective_var(col, q),
            label: q.question_name.clone(),
            group: q.group.clone(),
            qtype: q.qtype.clone(),
            max_points: q.max_points,
            position: q.position,
            estimate: q.estimate.clone(),
        })
        .collect();

    let mut students = Vec::new();
    for row in &rows {
        let rid = domain::row_id(row, &exam.id_columns);
        let mut points: HashMap<String, Option<f64>> = HashMap::new();
        for col in &exam.output_columns {
            if let Some(q) = exam.questions.get(col) {
                let var = effective_var(col, q);
                let g = q
                    .graded_items
                    .iter()
                    .find(|gi| gi.row_id == rid)
                    .and_then(|gi| crate::grade::evaluate_grade(&gi.grade));
                points.insert(var, g);
            }
        }
        let sr = scheme::compute_student(&rid, &scheme, &questions, &points);
        let est: std::collections::HashSet<&str> = sr.estimated.iter().map(|s| s.as_str()).collect();

        let mut rq = Vec::new();
        for col in &exam.output_columns {
            if let Some(q) = exam.questions.get(col) {
                let var = effective_var(col, q);
                let response = row.get(&q.input_column).cloned().unwrap_or_default();
                let pts = q
                    .graded_items
                    .iter()
                    .find(|gi| gi.row_id == rid)
                    .and_then(|gi| crate::grade::evaluate_grade(&gi.grade));
                rq.push(RenderQuestion {
                    label: if q.question_name.is_empty() {
                        col.clone()
                    } else {
                        q.question_name.clone()
                    },
                    group: q.group.clone(),
                    qtype: q.qtype.clone(),
                    response,
                    points: pts,
                    max: q.max_points,
                    estimated: est.contains(var.as_str()),
                });
            }
        }
        students.push(RenderStudent {
            id: rid,
            grade: sr.grade,
            total: sr.total,
            questions: rq,
        });
    }
    Ok(Json(RenderData { exam: name, students }))
}

async fn export_results_pdf(
    State(s): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<ResultsPdfReq>,
) -> AppResult<Json<Value>> {
    if s.config.renderer_url.is_empty() {
        return Err(AppError::BadRequest(
            "results renderer not configured (set RENDERER_URL)".into(),
        ));
    }
    {
        let conn = s.db().await;
        if !store::exam_exists(&conn, &name).await? {
            return Err(AppError::NotFound(format!("exam '{name}' not found")));
        }
    }
    let url = format!("{}/render", s.config.renderer_url.trim_end_matches('/'));
    let body = json!({ "exam": name, "scanned_pdf": req.scanned_pdf });
    let resp = s
        .http
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|e| AppError::Upstream(e.to_string()))?;
    if !resp.status().is_success() {
        let st = resp.status();
        let t = resp.text().await.unwrap_or_default();
        return Err(AppError::Upstream(format!("renderer {st}: {t}")));
    }
    let v: Value = resp.json().await.map_err(|e| AppError::Upstream(e.to_string()))?;
    Ok(Json(v))
}

async fn get_conflicts(
    State(s): State<AppState>,
    Path(name): Path<String>,
) -> AppResult<Json<Vec<domain::GradeConflict>>> {
    let conn = s.db().await;
    Ok(Json(store::list_conflicts(&conn, &name).await?))
}

async fn resolve_conflict(
    State(s): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<ResolveReq>,
) -> AppResult<StatusCode> {
    let conn = s.db().await;
    if req.choose == "incoming" {
        let conflicts = store::list_conflicts(&conn, &name).await?;
        if let Some(c) = conflicts
            .iter()
            .find(|c| c.output_col == req.output_col && c.row_id == req.row_id)
        {
            let item = GradedItem {
                row_id: c.row_id.clone(),
                input_text: c.input_text.clone(),
                grade: c.incoming_grade.clone(),
                timestamp: store::now_iso(),
            };
            store::put_graded_item(&conn, &name, &c.output_col, &item, &c.incoming_source, "").await?;
        }
    }
    store::clear_conflicts_for(&conn, &name, &req.output_col, &req.row_id).await?;
    store::touch(&conn, &name).await?;
    Ok(StatusCode::NO_CONTENT)
}
