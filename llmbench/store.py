"""SQLite persistence with explicit schema versioning, migrations and short transactions."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .util import now_iso

SCHEMA_VERSION = 2

SCHEMA_V2 = """
CREATE TABLE IF NOT EXISTS runs (
    run_id                  TEXT PRIMARY KEY,
    run_group               TEXT,
    model_alias             TEXT,
    model_id                TEXT,
    api_url                 TEXT,
    model_config_identity   TEXT,
    suite                   TEXT,
    profile                 TEXT,
    dataset                 TEXT,
    execution_status        TEXT,
    validity_status         TEXT,
    status_reason           TEXT,
    phase                   TEXT,
    comparability           TEXT,
    started_at              TEXT,
    finished_at             TEXT,
    num_requested           INTEGER,
    num_succeeded           INTEGER,
    num_errored             INTEGER,
    incomplete              INTEGER,
    evalscope_version       TEXT,
    tool_version            TEXT,
    git_commit              TEXT,
    output_dir              TEXT,
    report_path             TEXT,
    config_json             TEXT,
    protocol_identity       TEXT,
    sample_manifest_identity TEXT,
    primary_metric_json     TEXT,
    diagnostics_json        TEXT,
    perf_json               TEXT,
    error                   TEXT
);
CREATE TABLE IF NOT EXISTS metrics (
    run_id          TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    metric_key      TEXT NOT NULL,
    metric_name     TEXT NOT NULL,
    aggregation     TEXT,
    dimensions_json TEXT,
    category_key    TEXT NOT NULL,
    category_json   TEXT,
    subset          TEXT NOT NULL,
    num             INTEGER,
    score           REAL,
    macro_score     REAL,
    semantics_kind  TEXT,
    display_kind    TEXT,
    direction       TEXT,
    unit            TEXT,
    is_primary      INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (run_id, metric_key, category_key, subset)
);
CREATE TABLE IF NOT EXISTS sample_manifest (
    run_id          TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    subset          TEXT NOT NULL,
    selected        INTEGER,
    predicted       INTEGER,
    reviewed        INTEGER,
    sample_ids_json TEXT,
    question_digest TEXT,
    media_digest    TEXT,
    input_digest    TEXT,
    media_evidence  INTEGER,
    PRIMARY KEY (run_id, subset)
);
"""

RUN_INSERT_COLUMNS = (
    'run_id', 'run_group', 'model_alias', 'model_id', 'api_url', 'model_config_identity',
    'suite', 'profile', 'dataset', 'execution_status', 'validity_status', 'status_reason',
    'phase', 'comparability', 'started_at', 'evalscope_version', 'tool_version', 'git_commit',
    'output_dir', 'config_json', 'protocol_identity', 'sample_manifest_identity',
)


class DuplicateAttemptError(RuntimeError):
    """Raised when a run_id already exists; a new attempt must get a new id."""


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row['name'] for row in conn.execute(f'PRAGMA table_info({table})')}


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute('PRAGMA journal_mode=WAL')
        self.conn.execute('PRAGMA busy_timeout=5000')
        self.conn.execute('PRAGMA foreign_keys=ON')
        self.migrate()

    # -- schema management --------------------------------------------------

    def _schema_version(self) -> int:
        row = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='meta'"
        ).fetchone()
        if row is None:
            return 0
        row = self.conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        return int(row['value']) if row else 0

    def migrate(self) -> None:
        has_runs = self._has_table('runs')
        old_runs = has_runs and 'execution_status' not in _table_columns(self.conn, 'runs')
        leftover_v1 = any(self._has_table(name) for name in ('runs_v1', 'metrics_v1', 'run_samples_v1'))
        v1_metrics = self._has_table('metrics') and 'metric' in _table_columns(self.conn, 'metrics')
        if old_runs or leftover_v1 or v1_metrics:
            self._migrate_v1_to_v2()
        self.conn.executescript(
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);"
        )
        self.conn.executescript(SCHEMA_V2)
        self.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self.conn.commit()

    def _migrate_v1_to_v2(self) -> None:
        """Preserve v1 rows as legacy/unverified evidence; never fabricate identity.

        The rename-then-copy order is resumable: if a previous attempt died midway,
        leftover ``*_v1`` tables are picked up on the next open instead of being lost.
        """
        conn = self.conn
        if self._has_table('runs') and 'execution_status' not in _table_columns(conn, 'runs'):
            conn.execute('ALTER TABLE runs RENAME TO runs_v1')
        if self._has_table('metrics') and 'metric' in _table_columns(conn, 'metrics') \
                and not self._has_table('metrics_v1'):
            conn.execute('ALTER TABLE metrics RENAME TO metrics_v1')
        if self._has_table('run_samples') and not self._has_table('run_samples_v1'):
            conn.execute('ALTER TABLE run_samples RENAME TO run_samples_v1')
        conn.executescript(
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);"
        )
        conn.executescript(SCHEMA_V2)

        if self._has_table('runs_v1'):
            for row in conn.execute('SELECT * FROM runs_v1'):
                execution_status = 'completed' if row['status'] == 'success' else str(row['status'])
                validity_status = 'legacy' if row['status'] == 'success' else 'invalid'
                conn.execute(
                    """INSERT OR IGNORE INTO runs (run_id, run_group, model_alias, suite, profile,
                           dataset, execution_status, validity_status, status_reason, phase,
                           comparability, started_at, finished_at, num_requested, num_succeeded,
                           num_errored, incomplete, evalscope_version, tool_version, git_commit,
                           output_dir, config_json, error)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'legacy', 'legacy', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (row['run_id'], row['run_group'], row['model'], row['suite'], row['profile'],
                     row['dataset'], execution_status, validity_status, 'migrated from v1',
                     row['started_at'], row['finished_at'], row['num_requested'], row['num_succeeded'],
                     row['num_errored'], row['incomplete'], row['evalscope_version'], '0.1.0',
                     row['git_commit'], row['output_dir'], row['config_json'], row['error']),
                )
            conn.execute('DROP TABLE runs_v1')

        if self._has_table('metrics_v1'):
            for row in conn.execute('SELECT * FROM metrics_v1'):
                conn.execute(
                    """INSERT OR IGNORE INTO metrics (run_id, metric_key, metric_name, category_key,
                           category_json, subset, num, score, macro_score)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (row['run_id'], f"legacy:{row['metric']}", row['metric'],
                     f"legacy:{row['category']}", json.dumps([row['category']]),
                     row['subset'], row['num'], row['score'], row['macro_score']),
                )
            conn.execute('DROP TABLE metrics_v1')

        if self._has_table('run_samples_v1'):
            for row in conn.execute('SELECT * FROM run_samples_v1'):
                try:
                    ids = json.loads(row['sample_ids']) if row['sample_ids'] else None
                except json.JSONDecodeError:
                    ids = None
                count = None
                if isinstance(ids, dict):
                    count = sum(len(v) for v in ids.values())
                elif isinstance(ids, list):
                    count = len(ids)
                conn.execute(
                    """INSERT OR IGNORE INTO sample_manifest (run_id, subset, selected,
                           sample_ids_json, media_evidence)
                       VALUES (?, 'legacy', ?, ?, 0)""",
                    (row['run_id'], count, row['sample_ids']),
                )
            conn.execute('DROP TABLE run_samples_v1')
        conn.commit()

    def _has_table(self, name: str) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone() is not None

    # -- transactions -------------------------------------------------------

    @contextmanager
    def transaction(self):
        try:
            with self.conn:
                yield self.conn
        except Exception:
            raise

    # -- writes -------------------------------------------------------------

    def start_attempt(self, record: dict) -> None:
        columns = [c for c in RUN_INSERT_COLUMNS if c in record]
        placeholders = ', '.join('?' for _ in columns)
        values = [record[c] for c in columns]
        try:
            with self.transaction():
                self.conn.execute(
                    f"INSERT INTO runs ({', '.join(columns)}) VALUES ({placeholders})", values
                )
        except sqlite3.IntegrityError as exc:
            raise DuplicateAttemptError(f'run_id already exists: {record.get("run_id")}') from exc

    def finish_attempt(self, run_id: str, outcome: dict) -> None:
        """Atomically replace the attempt's metrics/manifest and write its final state."""
        with self.transaction():
            self.conn.execute(
                """UPDATE runs SET execution_status=?, validity_status=?, status_reason=?,
                       phase=?, comparability=?, finished_at=?, num_requested=?, num_succeeded=?,
                       num_errored=?, incomplete=?, report_path=?, diagnostics_json=?,
                       perf_json=?, error=?, protocol_identity=COALESCE(?, protocol_identity),
                       sample_manifest_identity=COALESCE(?, sample_manifest_identity),
                       primary_metric_json=?
                   WHERE run_id=?""",
                (outcome.get('execution_status'), outcome.get('validity_status'),
                 outcome.get('status_reason'), outcome.get('phase', 'done'),
                 outcome.get('comparability'), outcome.get('finished_at', now_iso()),
                 outcome.get('num_requested'), outcome.get('num_succeeded'),
                 outcome.get('num_errored'), outcome.get('incomplete'),
                 outcome.get('report_path'), _dumps(outcome.get('diagnostics')),
                 _dumps(outcome.get('perf_metrics')), outcome.get('error'),
                 outcome.get('protocol_identity'), outcome.get('sample_manifest_identity'),
                 _dumps(outcome.get('primary_metric_identity')),
                 run_id),
            )
            self.conn.execute('DELETE FROM metrics WHERE run_id=?', (run_id,))
            self.conn.executemany(
                """INSERT INTO metrics (run_id, metric_key, metric_name, aggregation, dimensions_json,
                       category_key, category_json, subset, num, score, macro_score, semantics_kind,
                       display_kind, direction, unit, is_primary)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [_metric_values(run_id, row) for row in outcome.get('metrics') or []],
            )
            self.conn.execute('DELETE FROM sample_manifest WHERE run_id=?', (run_id,))
            self.conn.executemany(
                """INSERT INTO sample_manifest (run_id, subset, selected, predicted, reviewed,
                       sample_ids_json, question_digest, media_digest, input_digest, media_evidence)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [_manifest_values(run_id, row) for row in outcome.get('sample_manifest') or []],
            )

    # -- reads --------------------------------------------------------------

    def get_attempt(self, run_id: str):
        return self.conn.execute('SELECT * FROM runs WHERE run_id=?', (run_id,)).fetchone()

    def attempt_exists(self, run_id: str) -> bool:
        return self.get_attempt(run_id) is not None

    def latest_extra(self, run_id: str):
        manifest = self.conn.execute(
            'SELECT * FROM sample_manifest WHERE run_id=? ORDER BY subset', (run_id,)
        ).fetchall()
        return manifest

    def close(self) -> None:
        self.conn.close()


def _dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True) if value is not None else None


def _metric_values(run_id: str, row: dict):
    return (
        run_id, row['metric_key'], row['metric_name'], row.get('aggregation'),
        _dumps(row.get('dimensions')), row['category_key'], _dumps(row.get('category')),
        row['subset'], row.get('num'), row.get('score'), row.get('macro_score'),
        row.get('semantics_kind'), row.get('display_kind'), row.get('direction'),
        row.get('unit'), 1 if row.get('is_primary') else 0,
    )


def _manifest_values(run_id: str, row: dict):
    return (
        run_id, row['subset'], row.get('selected'), row.get('predicted'), row.get('reviewed'),
        _dumps(row.get('sample_ids')), row.get('question_digest'), row.get('media_digest'),
        row.get('input_digest'), 1 if row.get('media_evidence') else 0,
    )
