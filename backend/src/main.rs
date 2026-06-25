//! Tentanator backend: the grading model exposed as an Axum HTTP API.

mod config;
mod domain;
mod error;
mod grade;
mod llm;
mod routes;
mod sampling;
mod store;
mod workspace;

use std::sync::Arc;

use config::Config;

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    pub http: reqwest::Client,
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

    // Import the oldest single-file session format if present.
    store::migrate_legacy_single_session(&config);

    if config.openai_api_key.is_empty() {
        tracing::warn!("OPENAI_API_KEY is empty - embeddings (maximin sampling) will fail");
    }
    if config.cerebras_api_key.is_empty() {
        tracing::warn!("CEREBRAS_API_KEY is empty - AI grade suggestions will fail");
    }

    let state = AppState {
        config: Arc::new(config),
        http: reqwest::Client::new(),
    };

    let app = routes::router(state);
    let listener = tokio::net::TcpListener::bind(&bind).await?;
    tracing::info!("tentanator backend listening on http://{bind}");
    axum::serve(listener, app).await?;
    Ok(())
}
