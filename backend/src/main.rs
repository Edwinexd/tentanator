//! Tentanator backend: the grading model exposed as an Axum HTTP API.

mod config;
mod db;
mod domain;
mod error;
mod grade;
mod llm;
mod routes;
mod sampling;
mod scheme;
mod store;
mod workspace;

use std::sync::Arc;

use config::Config;

/// Shared state. DB access goes through a single connection behind an async
/// mutex: Turso 0.4 errors on concurrent multi-connection writes, so we
/// serialize DB work (held only across DB calls, released across LLM/file I/O).
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    pub http: reqwest::Client,
    pub db: Arc<turso::Database>,
    pub conn: Arc<tokio::sync::Mutex<turso::Connection>>,
}

impl AppState {
    pub async fn db(&self) -> tokio::sync::MutexGuard<'_, turso::Connection> {
        self.conn.lock().await
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "tentanator_backend=info,tower_http=info".into()),
        )
        .init();

    let config = Config::from_env()?;
    let bind = config.bind_addr.clone();

    // Open the Turso database and import any legacy on-disk data into it.
    std::fs::create_dir_all(&config.data_dir).ok();
    let db_path = config.data_dir.join(".tentanator.db");
    let database = db::open(
        db_path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("data dir path is not valid UTF-8"))?,
    )
    .await?;
    let conn = database.connect()?;
    db::init_schema(&conn).await?;
    store::import_legacy_on_startup(&conn, &config).await?;

    if config.openai_api_key.is_empty() {
        tracing::warn!("OPENAI_API_KEY is empty - embeddings (maximin sampling) will fail");
    }
    if config.cerebras_api_key.is_empty() {
        tracing::warn!("CEREBRAS_API_KEY is empty - AI grade suggestions will fail");
    }

    let state = AppState {
        config: Arc::new(config),
        http: reqwest::Client::new(),
        db: Arc::new(database),
        conn: Arc::new(tokio::sync::Mutex::new(conn)),
    };

    let app = routes::router(state);
    let listener = tokio::net::TcpListener::bind(&bind).await?;
    tracing::info!("tentanator backend listening on http://{bind}");
    axum::serve(listener, app).await?;
    Ok(())
}
