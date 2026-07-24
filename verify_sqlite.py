import sqlite3
import tempfile
import os

db_path = tempfile.mktemp(suffix=".db")

conn = sqlite3.connect(db_path)
conn.execute("PRAGMA foreign_keys = ON")

# Create tables
conn.execute("""
CREATE TABLE eval_runs (
    run_id TEXT PRIMARY KEY
)
""")

conn.execute("""
CREATE TABLE case_results (
    case_id TEXT,
    run_id TEXT,
    FOREIGN KEY(run_id) REFERENCES eval_runs(run_id)
)
""")

print("Foreign Keys:", conn.execute("PRAGMA foreign_keys").fetchone()[0])

# Valid insert
conn.execute(
    "INSERT INTO eval_runs VALUES (?)",
    ("run-001",),
)

conn.execute(
    "INSERT INTO case_results VALUES (?, ?)",
    ("case-001", "run-001"),
)

print("✓ Valid foreign key insert succeeded")

# Invalid insert
try:
    conn.execute(
        "INSERT INTO case_results VALUES (?, ?)",
        ("case-002", "does-not-exist"),
    )
    conn.commit()
    print("❌ FAILED: Invalid foreign key was accepted")
except sqlite3.IntegrityError:
    print("✓ PASS: Foreign key constraint enforced")

conn.close()
os.remove(db_path)