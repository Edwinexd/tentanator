//! Core domain types. JSON shapes mirror the legacy Python session format so
//! existing `.tentanator_sessions/*.json` files stay readable.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// All fields default so partial / legacy session JSON never fails to load.
// Old sessions may carry extra fields (e.g. a per-item `embedding`); serde
// ignores unknown fields by default, so those import cleanly too.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GradedItem {
    #[serde(default)]
    pub row_id: String,
    #[serde(default)]
    pub input_text: String,
    #[serde(default)]
    pub grade: String,
    #[serde(default)]
    pub timestamp: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SamplingResult {
    #[serde(default)]
    pub algorithm: String,
    #[serde(default)]
    pub selected_ids: Vec<String>,
    #[serde(default)]
    pub quality_score: f64,
    #[serde(default)]
    pub num_samples: usize,
    #[serde(default)]
    pub timestamp: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct QuestionGrades {
    #[serde(default)]
    pub question_name: String,
    #[serde(default)]
    pub input_column: String,
    #[serde(default)]
    pub exam_question: String,
    #[serde(default)]
    pub sample_answer: String,
    #[serde(default)]
    pub global_question_id: Option<String>,
    #[serde(default)]
    pub graded_items: Vec<GradedItem>,
    #[serde(default)]
    pub sampling_result: Option<SamplingResult>,
    // --- examination grade-scheme config (per question) ---
    /// Identifier used in scheme expressions (e.g. "q1"); defaults derived from
    /// position when unset.
    #[serde(default)]
    pub var: String,
    /// Optional section/grouping tag (empty = ungrouped).
    #[serde(default)]
    pub group: String,
    /// Free-form type tag (mc / essay / ...).
    #[serde(default)]
    pub qtype: String,
    #[serde(default)]
    pub max_points: f64,
    #[serde(default)]
    pub position: i64,
    /// Optional expression to estimate points when ungraded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimate: Option<String>,
    /// Cross-session pooled items. Not persisted in the session file; hydrated
    /// from `global_bank/graded_pool/<gq_id>.jsonl` on load.
    #[serde(default, skip_serializing)]
    pub external_graded_items: Vec<GradedItem>,
}

impl QuestionGrades {
    /// Own grades merged with pooled ones (own wins on row_id collision).
    pub fn icl_candidates(&self) -> Vec<&GradedItem> {
        let own: std::collections::HashSet<&str> =
            self.graded_items.iter().map(|i| i.row_id.as_str()).collect();
        let mut merged: Vec<&GradedItem> = self.graded_items.iter().collect();
        for item in &self.external_graded_items {
            if !own.contains(item.row_id.as_str()) {
                merged.push(item);
            }
        }
        merged
    }

    pub fn valid_graded_count(&self) -> usize {
        self.graded_items
            .iter()
            .filter(|i| is_meaningful(&i.input_text))
            .count()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Session {
    #[serde(default)]
    pub session_name: String,
    pub csv_file: String,
    pub id_columns: Vec<String>,
    pub input_columns: Vec<String>,
    pub output_columns: Vec<String>,
    /// Optional grouping label (e.g. a course code). Replaces the old workspace
    /// directory hack - sessions are filtered/grouped by this, not by location.
    #[serde(default)]
    pub course: Option<String>,
    #[serde(default)]
    pub last_updated: String,
    #[serde(default)]
    pub questions: HashMap<String, QuestionGrades>,
    /// Examination grade scheme (vars + guarded band rules). None = unconfigured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scheme: Option<crate::scheme::GradeScheme>,
}

impl Session {
    /// Ensure a question exists for `output_col`, pairing it with the input
    /// column by index (falling back to the first input column).
    pub fn ensure_question(&mut self, output_col: &str) -> &mut QuestionGrades {
        if !self.questions.contains_key(output_col) {
            let idx = self
                .output_columns
                .iter()
                .position(|c| c == output_col)
                .unwrap_or(0);
            let input_col = self
                .input_columns
                .get(idx)
                .or_else(|| self.input_columns.first())
                .cloned()
                .unwrap_or_default();
            self.questions.insert(
                output_col.to_string(),
                QuestionGrades {
                    question_name: output_col.to_string(),
                    input_column: input_col,
                    ..Default::default()
                },
            );
        }
        self.questions.get_mut(output_col).unwrap()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AIGradeSuggestion {
    pub grade: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_summary: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SessionSummary {
    pub session_name: String,
    pub csv_file: String,
    pub course: Option<String>,
    pub last_updated: String,
    pub num_questions: usize,
    pub archived: bool,
}

/// An imported grade that disagrees with an existing one, pending resolution.
#[derive(Clone, Debug, Serialize)]
pub struct GradeConflict {
    pub output_col: String,
    pub row_id: String,
    pub existing_grade: String,
    pub existing_source: String,
    pub incoming_grade: String,
    pub incoming_source: String,
    pub input_text: String,
    pub timestamp: String,
}

/// Blank / dash / N/A responses are auto-zeroed and never count toward ICL.
pub fn is_meaningful(text: &str) -> bool {
    let t = text.trim();
    !(t.is_empty() || t == "-" || t == "N/A")
}

/// Build a row identifier by joining the ID column values with `_`.
pub fn row_id(row: &HashMap<String, String>, id_columns: &[String]) -> String {
    id_columns
        .iter()
        .map(|c| row.get(c).cloned().unwrap_or_default())
        .collect::<Vec<_>>()
        .join("_")
}
