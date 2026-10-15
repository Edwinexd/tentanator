//! "Value at export" state column for round-tripped sheets.
//!
//! The graded export and the second-pass review sheet end with a
//! [`COL_STATE`] column: per row, base64 JSON of the exam, the row id and the
//! grade cells as they were when the sheet was written. On import that gives a
//! three-way merge (export base, current store, sheet) instead of a two-way
//! one, so an untouched cell is never a conflict, an edited cell whose stored
//! value has not moved since export simply applies, and only a cell changed on
//! both sides is a genuine conflict. It also identifies the row and its
//! columns, so a sheet carrying it needs no manual column mapping. Sheets
//! without it (older exports, foreign files) keep the two-way behaviour.

use std::collections::{BTreeMap, HashMap};

use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use serde::{Deserialize, Serialize};

use crate::grade::grades_equal;

/// Header of the trailing state column.
pub const COL_STATE: &str = "_tentanator_state";
const VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RowState {
    pub v: u32,
    pub exam: String,
    pub id: String,
    /// Sheet column -> cell value at export.
    pub cells: BTreeMap<String, String>,
}

impl RowState {
    pub fn new(exam: &str, id: &str, cells: BTreeMap<String, String>) -> Self {
        RowState { v: VERSION, exam: exam.into(), id: id.into(), cells }
    }

    pub fn encode(&self) -> String {
        STANDARD.encode(serde_json::to_vec(self).unwrap_or_default())
    }

    pub fn decode(s: &str) -> Option<Self> {
        let bytes = STANDARD.decode(s.trim()).ok()?;
        serde_json::from_slice::<Self>(&bytes).ok().filter(|st| st.v == VERSION)
    }

    /// The state of an import row, if it carries one written for `exam`.
    pub fn of_row(row: &HashMap<String, String>, exam: &str) -> Option<Self> {
        Self::decode(row.get(COL_STATE)?).filter(|st| st.exam == exam)
    }

    pub fn base(&self, col: &str) -> Option<&str> {
        self.cells.get(col).map(String::as_str)
    }
}

/// True when any row of the sheet carries state for `exam`.
pub fn sheet_has_state(rows: &[HashMap<String, String>], exam: &str) -> bool {
    rows.iter().any(|r| RowState::of_row(r, exam).is_some())
}

/// What an import should do with one incoming cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Merge {
    /// Nothing stored yet: write it.
    New,
    /// Same as stored, or untouched in the sheet since export: leave it.
    Same,
    /// Edited in the sheet and the store has not moved since export: overwrite.
    Update,
    /// Stored value differs and cannot be reconciled: queue a conflict.
    Conflict,
}

/// Decide a cell. `base` is the value at export (`None` for a sheet without
/// state, which falls back to the two-way rule); `current` is the stored grade
/// (`""` when ungraded).
pub fn merge(base: Option<&str>, current: &str, incoming: &str) -> Merge {
    if grades_equal(current, incoming) {
        return Merge::Same;
    }
    match base {
        Some(b) if grades_equal(b, incoming) => Merge::Same,
        _ if current.trim().is_empty() => Merge::New,
        Some(b) if grades_equal(b, current) => Merge::Update,
        _ => Merge::Conflict,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_and_exam_filter() {
        let st = RowState::new("ex", "s1", BTreeMap::from([("Q1".into(), "2.5".into())]));
        let enc = st.encode();
        assert_eq!(RowState::decode(&enc), Some(st.clone()));
        let row = HashMap::from([(COL_STATE.to_string(), enc)]);
        assert!(RowState::of_row(&row, "ex").is_some());
        assert!(RowState::of_row(&row, "other").is_none());
        assert!(RowState::decode("not base64!").is_none());
        assert!(!sheet_has_state(&[HashMap::new()], "ex"));
    }

    #[test]
    fn three_way_merge() {
        // Untouched in the sheet, store moved since: not a conflict.
        assert_eq!(merge(Some("2"), "3", "2"), Merge::Same);
        // Edited in the sheet, store unchanged: overwrite.
        assert_eq!(merge(Some("2"), "1+1", "3"), Merge::Update);
        // Both sides changed differently: genuine conflict.
        assert_eq!(merge(Some("2"), "3", "4"), Merge::Conflict);
        // Both sides made the same change.
        assert_eq!(merge(Some("2"), "4", "4"), Merge::Same);
        // Ungraded at export and now, filled in the sheet.
        assert_eq!(merge(Some(""), "", "4"), Merge::New);
        // Graded since export while the sheet filled it too: conflict.
        assert_eq!(merge(Some(""), "3", "4"), Merge::Conflict);
        // No state: the old two-way rule.
        assert_eq!(merge(None, "", "4"), Merge::New);
        assert_eq!(merge(None, "4", "4"), Merge::Same);
        assert_eq!(merge(None, "3", "4"), Merge::Conflict);
    }
}
