//! HTTP routes. DB work goes through the shared mutexed connection
//! (`s.db().await`), held only across DB calls and released across LLM/file I/O.
//! All business logic lives in `store`, `sampling`, `grade`, `llm`.

use std::collections::HashMap;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::routing::{delete, get, post, put};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tower_http::cors::CorsLayer;

use crate::config::Config;
use crate::domain::{self, AIGradeSuggestion, GradedItem, QuestionGrades, Session, SessionSummary};
use crate::error::{AppError, AppResult};
use crate::grade::validate_grade;
use crate::sampling::{self, Algorithm};
use crate::scheme::{self, GradeScheme, StudentResult};
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
        .route("/api/sessions/{name}/scheme", put(put_scheme))
        .route("/api/sessions/{name}/questions-config", put(put_questions_config))
        .route("/api/sessions/{name}/results", get(get_results).post(preview_results))
        .route("/api/sessions/{name}/import/preview", post(import_preview))
        .route("/api/sessions/{name}/import/apply", post(import_apply))
        .route("/api/sessions/{name}/conflicts", get(get_conflicts))
        .route("/api/sessions/{name}/conflicts/resolve", post(resolve_conflict))
        .route("/api/sessions/{name}/export", post(export))
        .route("/api/sessions/{name}/export/daisy", post(export_daisy_route))
        .route("/api/sessions/{name}/export/csv", post(export_csv_route))
        .route("/api/sessions/{name}/render-data", get(render_data))
        .route("/api/sessions/{name}/export/results-pdf", post(export_results_pdf))
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

async fn load_session_or_404(s: &AppState, name: &str) -> AppResult<Session> {
    let conn = s.db().await;
    store::load_session(&conn, name)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("session '{name}' not found")))
}

fn no_question(col: &str) -> AppError {
    AppError::NotFound(format!("question '{col}' not found"))
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
    let conn = s.db().await;
    Ok(Json(workspace::import_workspace(&conn, &s.config, &name).await?))
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
) -> AppResult<Json<Vec<SessionSummary>>> {
    let mut sessions = {
        let conn = s.db().await;
        store::list_sessions(&conn, q.archived).await?
    };
    if let Some(course) = q.course.filter(|c| !c.is_empty()) {
        sessions.retain(|x| x.course.as_deref() == Some(course.as_str()));
    }
    Ok(Json(sessions))
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
    let raw_name = req.name.clone().unwrap_or_else(|| {
        let stem = std::path::Path::new(&req.csv_file)
            .file_stem()
            .and_then(|x| x.to_str())
            .unwrap_or("session");
        format!("{stem}_{}", store::timestamp_compact())
    });

    let mut session = Session {
        session_name: store::sanitize_name(&raw_name),
        csv_file: req.csv_file,
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
        session.ensure_question(col);
    }
    {
        let conn = s.db().await;
        store::insert_session(&conn, &session).await?;
    }
    Ok(Json(session))
}

async fn get_session(
    State(s): State<AppState>,
    Path(name): Path<String>,
) -> AppResult<Json<Session>> {
    Ok(Json(load_session_or_404(&s, &name).await?))
}

async fn update_session(
    State(s): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<SessionMeta>,
) -> AppResult<Json<Session>> {
    let conn = s.db().await;
    if !store::session_exists(&conn, &name).await? {
        return Err(AppError::NotFound(format!("session '{name}' not found")));
    }
    if let Some(course) = req.course {
        let course = if course.is_empty() { None } else { Some(course) };
        store::set_course(&conn, &name, course.as_deref()).await?;
    }
    store::touch(&conn, &name).await?;
    store::load_session(&conn, &name)
        .await?
        .map(Json)
        .ok_or_else(|| AppError::NotFound(format!("session '{name}' not found")))
}

async fn delete_session(
    State(s): State<AppState>,
    Path(name): Path<String>,
) -> AppResult<StatusCode> {
    let conn = s.db().await;
    if store::delete_session(&conn, &name).await? {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(AppError::NotFound(format!("session '{name}' not found")))
    }
}

async fn archive_session(
    State(s): State<AppState>,
    Path(name): Path<String>,
) -> AppResult<StatusCode> {
    let conn = s.db().await;
    if store::set_archived(&conn, &name, true).await? {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(AppError::NotFound(format!("session '{name}' not found")))
    }
}

async fn unarchive_session(
    State(s): State<AppState>,
    Path(name): Path<String>,
) -> AppResult<StatusCode> {
    let conn = s.db().await;
    if store::set_archived(&conn, &name, false).await? {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(AppError::NotFound(format!("session '{name}' not found")))
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
    let mut session = load_session_or_404(&s, &name).await?;
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
    let q = session.questions.get(&col).cloned().ok_or_else(|| no_question(&col))?;
    let conn = s.db().await;
    store::upsert_question_row(&conn, &name, &col, &q).await?;
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
    let mut session = load_session_or_404(&s, &name).await?;
    session.ensure_question(&col);
    let input_column = session.questions[&col].input_column.clone();
    let id_columns = session.id_columns.clone();
    let (rows, _) = session_exam(&s.config, &session)?;

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
        let q = session.ensure_question(&col);
        q.sampling_result = Some(result.clone());
    }
    let q = session.questions.get(&col).cloned().ok_or_else(|| no_question(&col))?;
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
    let mut session = load_session_or_404(&s, &name).await?;
    session.ensure_question(&col);
    let input_column = session.questions[&col].input_column.clone();
    let id_columns = session.id_columns.clone();

    let (rows, _) = session_exam(&s.config, &session)?;
    let input_text = rows
        .iter()
        .find(|r| domain::row_id(r, &id_columns) == req.row_id)
        .and_then(|r| r.get(&input_column).cloned())
        .unwrap_or_default();

    // Persist the question row first so the pool sync can read its global id.
    let q = session.questions.get(&col).cloned().ok_or_else(|| no_question(&col))?;
    {
        let conn = s.db().await;
        store::upsert_question_row(&conn, &name, &col, &q).await?;
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
    store::put_graded_item(&conn, &name, &col, &item, "manual").await?;
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
    let session = load_session_or_404(&s, &name).await?;
    let question = session.questions.get(&col).ok_or_else(|| no_question(&col))?;

    let id_columns = session.id_columns.clone();
    let input_column = question.input_column.clone();
    let (rows, _) = session_exam(&s.config, &session)?;
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
    let session = load_session_or_404(&s, &name).await?;
    let q = session.questions.get(&col).ok_or_else(|| no_question(&col))?;
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
    let session = load_session_or_404(&s, &name).await?;
    let (rows, cols) = session_exam(&s.config, &session)?;
    let path = store::export_to_excel(&s.config, &session, &rows, &cols)?;
    Ok(Json(json!({ "path": path })))
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
    position: i64,
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
    session: &Session,
    rows: &[HashMap<String, String>],
    scheme: &GradeScheme,
) -> ResultsResponse {
    let questions: Vec<scheme::QuestionConfig> = session
        .output_columns
        .iter()
        .filter_map(|col| session.questions.get(col).map(|q| (col, q)))
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
        let rid = domain::row_id(row, &session.id_columns);
        let mut points: HashMap<String, Option<f64>> = HashMap::new();
        for col in &session.output_columns {
            if let Some(q) = session.questions.get(col) {
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

async fn put_scheme(
    State(s): State<AppState>,
    Path(name): Path<String>,
    Json(scheme): Json<GradeScheme>,
) -> AppResult<StatusCode> {
    let conn = s.db().await;
    if !store::session_exists(&conn, &name).await? {
        return Err(AppError::NotFound(format!("session '{name}' not found")));
    }
    store::set_scheme(&conn, &name, &Some(scheme)).await?;
    store::touch(&conn, &name).await?;
    Ok(StatusCode::NO_CONTENT)
}

async fn put_questions_config(
    State(s): State<AppState>,
    Path(name): Path<String>,
    Json(updates): Json<Vec<QuestionConfigUpdate>>,
) -> AppResult<Json<Session>> {
    let mut session = load_session_or_404(&s, &name).await?;
    {
        let conn = s.db().await;
        for u in &updates {
            {
                let q = session.ensure_question(&u.col);
                q.var = u.var.clone();
                q.group = u.group.clone();
                q.qtype = u.qtype.clone();
                q.max_points = u.max_points;
                q.position = u.position;
                q.estimate = u.estimate.clone().filter(|e| !e.is_empty());
            }
            let qc = session.questions.get(&u.col).cloned().ok_or_else(|| no_question(&u.col))?;
            store::upsert_question_row(&conn, &name, &u.col, &qc).await?;
        }
        store::touch(&conn, &name).await?;
    }
    Ok(Json(load_session_or_404(&s, &name).await?))
}

async fn get_results(
    State(s): State<AppState>,
    Path(name): Path<String>,
) -> AppResult<Json<ResultsResponse>> {
    let session = load_session_or_404(&s, &name).await?;
    let (rows, _) = session_exam(&s.config, &session)?;
    let mut resp = match &session.scheme {
        Some(scheme) => compute_results(&session, &rows, scheme),
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
    let session = load_session_or_404(&s, &name).await?;
    let (rows, _) = session_exam(&s.config, &session)?;
    Ok(Json(compute_results(&session, &rows, &scheme)))
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
        _ => a.trim() == b.trim(),
    }
}

/// Read roster + import file, returning (roster id-set, roster by id, import rows).
fn load_import(
    s: &AppState,
    session: &Session,
    file: &str,
) -> AppResult<(std::collections::HashSet<String>, Vec<HashMap<String, String>>)> {
    let (roster, _) = session_exam(&s.config, session)?;
    let ids: std::collections::HashSet<String> = roster
        .iter()
        .map(|r| domain::row_id(r, &session.id_columns))
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
    let session = store::load_session(&conn, &name)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("session '{name}' not found")))?;
    let (roster_ids, irows) = load_import(&s, &session, &req.file)?;

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
    let session = store::load_session(&conn, &name)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("session '{name}' not found")))?;
    let (roster, _) = session_exam(&s.config, &session)?;
    let roster_by_id: HashMap<String, &HashMap<String, String>> = roster
        .iter()
        .map(|r| (domain::row_id(r, &session.id_columns), r))
        .collect();
    let (roster_ids, irows) = load_import(&s, &session, &req.file)?;
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
            let input_text = session
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
                        summary.same += 1;
                    } else {
                        summary.new += 1;
                    }
                    let item = GradedItem {
                        row_id: id.clone(),
                        input_text,
                        grade: incoming.clone(),
                        timestamp: store::now_iso(),
                    };
                    store::put_graded_item(&conn, &name, &m.output_col, &item, &label).await?;
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
) -> AppResult<Json<Value>> {
    let session = load_session_or_404(&s, &name).await?;
    let (rows, _) = session_exam(&s.config, &session)?;
    let scheme = session
        .scheme
        .as_ref()
        .ok_or_else(|| AppError::BadRequest("no grade scheme configured".into()))?;
    let resp = compute_results(&session, &rows, scheme);
    let path = store::export_daisy(&s.config, &session, &resp.results)?;
    Ok(Json(json!({ "path": path })))
}

async fn export_csv_route(
    State(s): State<AppState>,
    Path(name): Path<String>,
) -> AppResult<Json<Value>> {
    let session = load_session_or_404(&s, &name).await?;
    let (rows, _) = session_exam(&s.config, &session)?;
    let results = session
        .scheme
        .as_ref()
        .map(|sc| compute_results(&session, &rows, sc).results)
        .unwrap_or_default();
    let path = store::export_per_question_csv(&s.config, &session, &rows, &results)?;
    Ok(Json(json!({ "path": path })))
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
    let session = load_session_or_404(&s, &name).await?;
    let (rows, _) = session_exam(&s.config, &session)?;
    let scheme = session.scheme.clone().unwrap_or_default();
    let questions: Vec<scheme::QuestionConfig> = session
        .output_columns
        .iter()
        .filter_map(|col| session.questions.get(col).map(|q| (col, q)))
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
        let rid = domain::row_id(row, &session.id_columns);
        let mut points: HashMap<String, Option<f64>> = HashMap::new();
        for col in &session.output_columns {
            if let Some(q) = session.questions.get(col) {
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
        for col in &session.output_columns {
            if let Some(q) = session.questions.get(col) {
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
        if !store::session_exists(&conn, &name).await? {
            return Err(AppError::NotFound(format!("session '{name}' not found")));
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
            store::put_graded_item(&conn, &name, &c.output_col, &item, &c.incoming_source).await?;
        }
    }
    store::clear_conflicts_for(&conn, &name, &req.output_col, &req.row_id).await?;
    store::touch(&conn, &name).await?;
    Ok(StatusCode::NO_CONTENT)
}
