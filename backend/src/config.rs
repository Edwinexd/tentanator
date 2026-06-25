//! Runtime configuration loaded from the environment (`.env` supported).

use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct Config {
    /// Root directory that holds exams/, graded_exams/, .tentanator_sessions/, etc.
    pub data_dir: PathBuf,
    pub openai_api_key: String,
    pub openai_base_url: String,
    pub embedding_model: String,
    pub cerebras_api_key: String,
    pub cerebras_base_url: String,
    /// Reasoning model used for grading inference.
    pub cerebras_model: String,
    pub cerebras_reasoning_effort: String,
    /// Model used to condense reasoning chains into a short summary.
    pub cerebras_summary_model: String,
    pub bind_addr: String,
}

impl Config {
    pub fn from_env() -> anyhow::Result<Self> {
        // Load .env from the data dir / cwd if present. Missing file is fine.
        let _ = dotenvy::dotenv();

        let data_dir = std::env::var("TENTANATOR_DATA_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("."));

        let openai_api_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
        let cerebras_api_key = std::env::var("CEREBRAS_API_KEY").unwrap_or_default();

        Ok(Self {
            data_dir,
            openai_api_key,
            openai_base_url: std::env::var("OPENAI_BASE_URL")
                .unwrap_or_else(|_| "https://api.openai.com/v1".to_string()),
            embedding_model: std::env::var("EMBEDDING_MODEL")
                .unwrap_or_else(|_| "text-embedding-3-large".to_string()),
            cerebras_api_key,
            cerebras_base_url: std::env::var("CEREBRAS_BASE_URL")
                .unwrap_or_else(|_| "https://api.cerebras.ai/v1".to_string()),
            cerebras_model: std::env::var("CEREBRAS_MODEL")
                .unwrap_or_else(|_| "gpt-oss-120b".to_string()),
            cerebras_reasoning_effort: std::env::var("CEREBRAS_REASONING_EFFORT")
                .unwrap_or_else(|_| "high".to_string()),
            cerebras_summary_model: std::env::var("CEREBRAS_SUMMARY_MODEL")
                .unwrap_or_else(|_| "gpt-oss-120b".to_string()),
            bind_addr: std::env::var("TENTANATOR_BIND")
                .unwrap_or_else(|_| "127.0.0.1:8787".to_string()),
        })
    }

    pub fn exams_dir(&self) -> PathBuf {
        self.data_dir.join("exams")
    }

    pub fn graded_dir(&self) -> PathBuf {
        self.data_dir.join("graded_exams")
    }

    pub fn sessions_dir(&self) -> PathBuf {
        self.data_dir.join(".tentanator_sessions")
    }

    pub fn archive_dir(&self) -> PathBuf {
        self.sessions_dir().join("archive")
    }

    pub fn graded_pool_dir(&self) -> PathBuf {
        self.data_dir.join("global_bank").join("graded_pool")
    }
}
