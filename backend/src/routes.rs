//! HTTP routes. Each handler is a thin orchestration layer over the domain
//! modules; all business logic lives in `store`, `sampling`, `grade`, `llm`.

use std::collections::HashMap;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::routing::{delete, get, post, put};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};
use tower_http::cors::CorsLayer;

use crate::config::Config;
use crate::domain::{self, AIGradeSuggestion, GradedItem, QuestionGrades, Session, SessionSummary};
use crate::error::{AppError, AppResult};
use crate::grade::validate_grade;
use crate::sampling::{self, Algorithm};
use crate::{llm, store, workspace, AppState};

const NUM_REPRESENTATIVE_SAMPLES: usize = 5;

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/api/health", get(health))
        .route("/api/legacy-workspaces", get(list_legacy_workspaces))
        .route("/api/legacy-workspaces/{name}/import", post(import_workspace))
        .route("/api/exams", get(list_exams))
        .route("/api/exams/{file}/columns", get(exam_columns))
        .route("/api/exams/{file}/rows", get(exam_rows))
        .route("/api/sessions", get(list_sessions).post(create_session))
        .route(
            "/api/sessions/{name}",
            get(get_session).put(update_session).delete(delete_session),
        )
        .route("/api/sessions/{name}/archive", post(archive_session))
        .route("/api/sessions/{name}/unarchive", post(unarchive_session))
        .route("/api/sessions/{name}/questions/{col}", put(put_question))
        .route("/api/sessions/{name}/questions/{col}/sampling", post(run_sampling))
        .route("/api/sessions/{name}/questions/{col}/grade", post(grade_item))
        .route(
            "/api/sessions/{name}/questions/{col}/grade/{row_id}",
            delete(ungrade_item),
        )
        .route("/api/sessions/{name}/questions/{col}/suggest", post(suggest))
        .route("/api/sessions/{name}/questions/{col}/status", get(question_status))
        .route("/api/sessions/{name}/export", post(export))
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
struct CreateSession {
    csv_file: String,
    id_columns: Vec<String>,
    input_columns: Vec<String>,
    output_columns: Vec<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    course: Option<String>,
}

#[derive(Deserialize)]
struct SessionMeta {
    course: Option<String>,
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
}

#[derive(Deserialize)]
struct SuggestReq {
    row_id: String,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Load exam rows + ordered columns for a session's csv_file.
fn session_exam(
    config: &Config,
    session: &Session,
) -> AppResult<(Vec<HashMap<String, String>>, Vec<String>)> {
    let path = store::resolve_exam_path(config, &session.csv_file).ok_or_else(|| {
        AppError::NotFound(format!("exam file '{}' not found", session.csv_file))
    })?;
    let rows = store::read_exam_data(&path)?;
    let cols = store::get_exam_columns(&path)?;
    Ok((rows, cols))
}

fn question_clone(session: &Session, col: &str) -> AppResult<QuestionGrades> {
    session
        .questions
        .get(col)
        .cloned()
        .ok_or_else(|| AppError::NotFound(format!("question '{col}' not found")))
}

// ---------------------------------------------------------------------------
// Health, legacy import & exams
// ---------------------------------------------------------------------------

async fn health() -> Json<Value> {
    Json(json!({ "status": "ok" }))
}

async fn list_legacy_workspaces(
    State(s): State<AppState>,
) -> Json<Vec<workspace::WorkspaceInfo>> {
    Json(workspace::list_importable(&s.config))
}

async fn import_workspace(
    State(s): State<AppState>,
    Path(name): Path<String>,
) -> AppResult<Json<workspace::ImportResult>> {
    Ok(Json(workspace::import_workspace(&s.config, &name)?))
}

async fn list_exams(State(s): State<AppState>) -> Json<Vec<String>> {
    Json(store::list_exam_files(&s.config))
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
// Sessions
// ---------------------------------------------------------------------------

async fn list_sessions(
    State(s): State<AppState>,
    Query(q): Query<ListQuery>,
) -> Json<Vec<SessionSummary>> {
    let mut sessions = store::list_sessions(&s.config, q.archived);
    if let Some(course) = q.course.filter(|c| !c.is_empty()) {
        sessions.retain(|x| x.course.as_deref() == Some(course.as_str()));
    }
    Json(sessions)
}

async fn create_session(
    State(s): State<AppState>,
    Json(req): Json<CreateSession>,
) -> AppResult<Json<Session>> {
    if req.input_columns.is_empty() || req.output_columns.is_empty() {
        return Err(AppError::BadRequest(
            "input_columns and output_columns are required".into(),
        ));
    }
    let name = req.name.clone().unwrap_or_else(|| {
        let stem = std::path::Path::new(&req.csv_file)
            .file_stem()
            .and_then(|x| x.to_str())
            .unwrap_or("session");
        format!("{stem}_{}", store::timestamp_compact())
    });

    let mut session = Session {
        session_name: String::new(),
        csv_file: req.csv_file,
        id_columns: if req.id_columns.is_empty() {
            vec!["_row_number".to_string()]
        } else {
            req.id_columns
        },
        input_columns: req.input_columns,
        output_columns: req.output_columns.clone(),
        course: req.course.filter(|c| !c.is_empty()),
        last_updated: String::new(),
        questions: HashMap::new(),
        embeddings_cache: HashMap::new(),
        features_cache: HashMap::new(),
    };
    for col in &req.output_columns {
        session.ensure_question(col);
    }
    store::save_session(&s.config, &mut session, &name)?;
    Ok(Json(session))
}

async fn get_session(
    State(s): State<AppState>,
    Path(name): Path<String>,
) -> AppResult<Json<Session>> {
    // Try active, then archived.
    match store::load_session(&s.config, &name, false) {
        Ok(sess) => Ok(Json(sess)),
        Err(_) => Ok(Json(store::load_session(&s.config, &name, true)?)),
    }
}

async fn update_session(
    State(s): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<SessionMeta>,
) -> AppResult<Json<Session>> {
    let mut session = store::load_session(&s.config, &name, false)?;
    if let Some(course) = req.course {
        session.course = if course.is_empty() { None } else { Some(course) };
    }
    store::save_session(&s.config, &mut session, &name)?;
    Ok(Json(session))
}

async fn delete_session(
    State(s): State<AppState>,
    Path(name): Path<String>,
) -> AppResult<StatusCode> {
    store::delete_session(&s.config, &name)?;
    Ok(StatusCode::NO_CONTENT)
}

async fn archive_session(
    State(s): State<AppState>,
    Path(name): Path<String>,
) -> AppResult<StatusCode> {
    store::set_archived(&s.config, &name, true)?;
    Ok(StatusCode::NO_CONTENT)
}

async fn unarchive_session(
    State(s): State<AppState>,
    Path(name): Path<String>,
) -> AppResult<StatusCode> {
    store::set_archived(&s.config, &name, false)?;
    Ok(StatusCode::NO_CONTENT)
}

// ---------------------------------------------------------------------------
// Questions & grading
// ---------------------------------------------------------------------------

async fn put_question(
    State(s): State<AppState>,
    Path((name, col)): Path<(String, String)>,
    Json(req): Json<QuestionMeta>,
) -> AppResult<Json<QuestionGrades>> {
    let mut session = store::load_session(&s.config, &name, false)?;
    {
        let q = session.ensure_question(&col);
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
    if let Some(q) = session.questions.get_mut(&col) {
        store::hydrate_external_graded_items(&s.config, q, Some(&name));
    }
    store::save_session(&s.config, &mut session, &name)?;
    Ok(Json(question_clone(&session, &col)?))
}

async fn run_sampling(
    State(s): State<AppState>,
    Path((name, col)): Path<(String, String)>,
    Json(req): Json<SamplingReq>,
) -> AppResult<Json<crate::domain::SamplingResult>> {
    let config = &s.config;
    let mut session = store::load_session(config, &name, false)?;
    session.ensure_question(&col);

    let input_column = session.questions[&col].input_column.clone();
    let id_columns = session.id_columns.clone();
    let (rows, _) = session_exam(config, &session)?;

    // Candidate responses (meaningful, non-blank), in file order.
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
        // Random needs no embeddings - just pick over candidate row ids.
        Algorithm::Random => {
            let ids: Vec<String> = candidates.iter().map(|(rid, _)| rid.clone()).collect();
            sampling::random_sample(&ids, n)
        }
        // Maximin needs embeddings; generate any that are missing and cache them.
        Algorithm::Maximin => {
            let existing: std::collections::HashSet<String> = session
                .features_cache
                .get(&input_column)
                .map(|m| m.keys().cloned().collect())
                .unwrap_or_default();
            let to_embed: Vec<(String, String)> = candidates
                .iter()
                .filter(|(rid, _)| !existing.contains(rid))
                .cloned()
                .collect();
            if !to_embed.is_empty() {
                let texts: Vec<String> = to_embed.iter().map(|(_, t)| t.clone()).collect();
                let embs = llm::embed_many(config, &s.http, &texts).await;
                let entry = session.features_cache.entry(input_column.clone()).or_default();
                for ((rid, _), emb) in to_embed.iter().zip(embs) {
                    if let Some(v) = emb {
                        entry.insert(rid.clone(), v);
                    }
                }
            }
            let cache = session.features_cache.get(&input_column);
            let valid: Vec<(String, Vec<f32>)> = candidates
                .iter()
                .filter_map(|(rid, _)| {
                    cache
                        .and_then(|m| m.get(rid))
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
    session.ensure_question(&col).sampling_result = Some(result.clone());
    store::save_session(config, &mut session, &name)?;
    Ok(Json(result))
}

async fn grade_item(
    State(s): State<AppState>,
    Path((name, col)): Path<(String, String)>,
    Json(req): Json<GradeReq>,
) -> AppResult<Json<QuestionGrades>> {
    let config = &s.config;
    validate_grade(&req.grade).map_err(AppError::BadRequest)?;

    let mut session = store::load_session(config, &name, false)?;
    session.ensure_question(&col);
    let input_column = session.questions[&col].input_column.clone();
    let id_columns = session.id_columns.clone();

    let (rows, _) = session_exam(config, &session)?;
    let input_text = rows
        .iter()
        .find(|r| domain::row_id(r, &id_columns) == req.row_id)
        .and_then(|r| r.get(&input_column).cloned())
        .unwrap_or_default();

    // Cache a feature vector so the row can participate in future sampling.
    if domain::is_meaningful(&input_text) {
        let cached = session
            .features_cache
            .get(&input_column)
            .map(|m| m.contains_key(&req.row_id))
            .unwrap_or(false);
        if !cached {
            if let Ok(v) = llm::embed(config, &s.http, &input_text).await {
                session
                    .features_cache
                    .entry(input_column.clone())
                    .or_default()
                    .insert(req.row_id.clone(), v);
            }
        }
    }

    let item = GradedItem {
        row_id: req.row_id.clone(),
        input_text,
        grade: req.grade.clone(),
        timestamp: store::now_iso(),
    };
    {
        let q = session.ensure_question(&col);
        q.graded_items.retain(|i| i.row_id != req.row_id);
        q.graded_items.push(item);
    }
    store::save_session(config, &mut session, &name)?;
    Ok(Json(question_clone(&session, &col)?))
}

async fn ungrade_item(
    State(s): State<AppState>,
    Path((name, col, row_id)): Path<(String, String, String)>,
) -> AppResult<Json<QuestionGrades>> {
    let mut session = store::load_session(&s.config, &name, false)?;
    if let Some(q) = session.questions.get_mut(&col) {
        q.graded_items.retain(|i| i.row_id != row_id);
    }
    store::save_session(&s.config, &mut session, &name)?;
    Ok(Json(question_clone(&session, &col)?))
}

async fn suggest(
    State(s): State<AppState>,
    Path((name, col)): Path<(String, String)>,
    Json(req): Json<SuggestReq>,
) -> AppResult<Json<AIGradeSuggestion>> {
    let config = &s.config;
    let session = store::load_session(config, &name, false)?;
    let question = session
        .questions
        .get(&col)
        .ok_or_else(|| AppError::NotFound(format!("question '{col}' not found")))?;

    let id_columns = session.id_columns.clone();
    let input_column = question.input_column.clone();
    let (rows, _) = session_exam(config, &session)?;
    let response_text = rows
        .iter()
        .find(|r| domain::row_id(r, &id_columns) == req.row_id)
        .and_then(|r| r.get(&input_column).cloned())
        .ok_or_else(|| AppError::NotFound(format!("row '{}' not found", req.row_id)))?;

    let suggestion = llm::suggest_grade(config, &s.http, question, &response_text).await?;
    Ok(Json(suggestion))
}

async fn question_status(
    State(s): State<AppState>,
    Path((name, col)): Path<(String, String)>,
) -> AppResult<Json<Value>> {
    let session = store::load_session(&s.config, &name, false)?;
    let q = session
        .questions
        .get(&col)
        .ok_or_else(|| AppError::NotFound(format!("question '{col}' not found")))?;
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
) -> AppResult<Json<Value>> {
    let config = &s.config;
    let session = store::load_session(config, &name, false)?;
    let (rows, cols) = session_exam(config, &session)?;
    let path = store::export_to_excel(config, &session, &rows, &cols)?;
    Ok(Json(json!({ "path": path })))
}
