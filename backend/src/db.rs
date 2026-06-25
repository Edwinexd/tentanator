//! Turso (the Rust SQLite rewrite) storage layer: connection + schema + helpers.
//!
//! Optional TEXT columns use an empty-string sentinel rather than NULL to keep
//! Turso parameter binding simple. Embedding vectors are stored as little-endian
//! f32 BLOBs.

use turso::Builder;

/// Open (and create if needed) a local Turso database.
pub async fn open(path: &str) -> turso::Result<turso::Database> {
    Builder::new_local(path).build().await
}

/// Create the schema if it does not already exist.
pub async fn init_schema(conn: &turso::Connection) -> turso::Result<()> {
    conn.execute(
        "CREATE TABLE IF NOT EXISTS sessions (
            name TEXT PRIMARY KEY,
            csv_file TEXT NOT NULL,
            id_columns TEXT NOT NULL,
            input_columns TEXT NOT NULL,
            output_columns TEXT NOT NULL,
            course TEXT NOT NULL DEFAULT '',
            last_updated TEXT NOT NULL DEFAULT '',
            archived INTEGER NOT NULL DEFAULT 0,
            scheme TEXT NOT NULL DEFAULT ''
        )",
        (),
    )
    .await?;

    conn.execute(
        "CREATE TABLE IF NOT EXISTS questions (
            session_name TEXT NOT NULL,
            output_col TEXT NOT NULL,
            question_name TEXT NOT NULL DEFAULT '',
            input_column TEXT NOT NULL DEFAULT '',
            exam_question TEXT NOT NULL DEFAULT '',
            sample_answer TEXT NOT NULL DEFAULT '',
            global_question_id TEXT NOT NULL DEFAULT '',
            sampling_result TEXT NOT NULL DEFAULT '',
            var TEXT NOT NULL DEFAULT '',
            qgroup TEXT NOT NULL DEFAULT '',
            qtype TEXT NOT NULL DEFAULT '',
            max_points REAL NOT NULL DEFAULT 0,
            position INTEGER NOT NULL DEFAULT 0,
            estimate TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (session_name, output_col)
        )",
        (),
    )
    .await?;

    conn.execute(
        "CREATE TABLE IF NOT EXISTS graded_items (
            session_name TEXT NOT NULL,
            output_col TEXT NOT NULL,
            row_id TEXT NOT NULL,
            input_text TEXT NOT NULL DEFAULT '',
            grade TEXT NOT NULL DEFAULT '',
            timestamp TEXT NOT NULL DEFAULT '',
            source TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (session_name, output_col, row_id)
        )",
        (),
    )
    .await?;

    conn.execute(
        "CREATE TABLE IF NOT EXISTS caches (
            session_name TEXT NOT NULL,
            kind TEXT NOT NULL,
            input_column TEXT NOT NULL,
            row_id TEXT NOT NULL,
            vector BLOB NOT NULL,
            PRIMARY KEY (session_name, kind, input_column, row_id)
        )",
        (),
    )
    .await?;

    conn.execute(
        "CREATE TABLE IF NOT EXISTS graded_pool (
            global_question_id TEXT NOT NULL,
            source_session TEXT NOT NULL,
            row_id TEXT NOT NULL,
            input_text TEXT NOT NULL DEFAULT '',
            grade TEXT NOT NULL DEFAULT '',
            timestamp TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (global_question_id, source_session, row_id)
        )",
        (),
    )
    .await?;

    conn.execute(
        "CREATE TABLE IF NOT EXISTS grade_conflicts (
            session_name TEXT NOT NULL,
            output_col TEXT NOT NULL,
            row_id TEXT NOT NULL,
            existing_grade TEXT NOT NULL DEFAULT '',
            existing_source TEXT NOT NULL DEFAULT '',
            incoming_grade TEXT NOT NULL DEFAULT '',
            incoming_source TEXT NOT NULL DEFAULT '',
            input_text TEXT NOT NULL DEFAULT '',
            timestamp TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (session_name, output_col, row_id, incoming_source)
        )",
        (),
    )
    .await?;

    // Best-effort column migrations for DBs created by an earlier schema.
    // Duplicate-column errors are expected and ignored.
    for stmt in [
        "ALTER TABLE questions ADD COLUMN var TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE questions ADD COLUMN qgroup TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE questions ADD COLUMN qtype TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE questions ADD COLUMN max_points REAL NOT NULL DEFAULT 0",
        "ALTER TABLE questions ADD COLUMN position INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE questions ADD COLUMN estimate TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE sessions ADD COLUMN scheme TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE graded_items ADD COLUMN source TEXT NOT NULL DEFAULT ''",
    ] {
        let _ = conn.execute(stmt, ()).await;
    }

    Ok(())
}

pub fn f32s_to_blob(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

pub fn blob_to_f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[cfg(test)]
mod spike {
    use super::*;

    // Confirms the Turso features the store relies on: composite primary keys,
    // BLOB columns, parameterized queries, and BEGIN/COMMIT.
    #[tokio::test]
    async fn turso_supports_what_we_need() -> turso::Result<()> {
        let dir = std::env::temp_dir().join(format!("tt-turso-spike-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("spike.db");
        let db = open(path.to_str().unwrap()).await?;
        let conn = db.connect()?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS t (a TEXT, b TEXT, v BLOB, PRIMARY KEY(a, b))",
            (),
        )
        .await?;
        conn.execute(
            "INSERT INTO t (a, b, v) VALUES (?, ?, ?)",
            ("s1", "c1", vec![1u8, 2, 3]),
        )
        .await?;
        conn.execute("DELETE FROM t WHERE a = ? AND b = ?", ("s1", "c1")).await?;
        conn.execute(
            "INSERT INTO t (a, b, v) VALUES (?, ?, ?)",
            ("s1", "c1", db_blob(&[4.0, 5.0])),
        )
        .await?;

        conn.execute("BEGIN", ()).await?;
        conn.execute(
            "INSERT INTO t (a, b, v) VALUES (?, ?, ?)",
            ("s2", "c1", Vec::<u8>::new()),
        )
        .await?;
        conn.execute("COMMIT", ()).await?;

        let mut rows = conn.query("SELECT a, v FROM t WHERE a = ?", ("s1",)).await?;
        let mut seen = 0;
        while let Some(row) = rows.next().await? {
            let a: String = row.get(0)?;
            let v: Vec<u8> = row.get(1)?;
            assert_eq!(a, "s1");
            assert_eq!(blob_to_f32s(&v), vec![4.0, 5.0]);
            seen += 1;
        }
        assert_eq!(seen, 1);

        let mut count = conn.query("SELECT COUNT(*) FROM t", ()).await?;
        let n: i64 = count.next().await?.unwrap().get(0)?;
        assert_eq!(n, 2);

        let _ = std::fs::remove_dir_all(&dir);
        Ok(())
    }

    fn db_blob(v: &[f32]) -> Vec<u8> {
        f32s_to_blob(v)
    }
}
