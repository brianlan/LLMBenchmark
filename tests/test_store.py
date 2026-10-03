import json
import sqlite3
import threading

import pytest

from llmbench.store import DuplicateAttemptError, Store

V1_SCHEMA = """
CREATE TABLE runs (
    run_id TEXT PRIMARY KEY, run_group TEXT, model TEXT, suite TEXT, profile TEXT,
    dataset TEXT, status TEXT, started_at TEXT, finished_at TEXT, num_requested INTEGER,
    num_succeeded INTEGER, num_errored INTEGER, incomplete INTEGER, evalscope_version TEXT,
    git_commit TEXT, output_dir TEXT, config_json TEXT, perf_json TEXT, error TEXT
);
CREATE TABLE metrics (
    run_id TEXT, dataset TEXT, metric TEXT, category TEXT, subset TEXT,
    num INTEGER, score REAL, macro_score REAL,
    PRIMARY KEY (run_id, metric, category, subset)
);
CREATE TABLE run_samples (
    run_id TEXT, dataset TEXT, sample_ids TEXT, PRIMARY KEY (run_id, dataset)
);
"""


def attempt_record(run_id='run-1'):
    return {
        'run_id': run_id, 'run_group': 'g', 'model_alias': 'minimax',
        'model_id': 'm', 'api_url': 'https://api.example/v1',
        'model_config_identity': 'mid', 'suite': 'knowledge', 'profile': 'lite',
        'dataset': 'gpqa_diamond', 'execution_status': 'running', 'validity_status': None,
        'status_reason': None, 'phase': 'preflight', 'comparability': 'unknown',
        'started_at': '2026-01-01T00:00:00', 'evalscope_version': '1.12.0',
        'tool_version': '0.2.0', 'git_commit': 'abc', 'output_dir': '/tmp/out',
        'config_json': '{}', 'protocol_identity': 'pid', 'sample_manifest_identity': None,
    }


def outcome(score=0.5):
    return {
        'execution_status': 'completed', 'validity_status': 'complete',
        'status_reason': 'complete', 'phase': 'persist', 'comparability': 'verified',
        'finished_at': '2026-01-01T00:01:00', 'num_requested': 5, 'num_succeeded': 5,
        'num_errored': 0, 'incomplete': False, 'report_path': '/tmp/out/report.json',
        'metrics': [{
            'metric_key': 'mk', 'metric_name': 'accuracy', 'aggregation': 'mean',
            'dimensions': {}, 'category_key': 'ck', 'category': [], 'subset': '',
            'num': 5, 'score': score, 'macro_score': None, 'semantics_kind': 'quality',
            'display_kind': 'percent', 'direction': 'higher', 'unit': '%', 'is_primary': True,
        }],
        'sample_manifest': [{
            'subset': 'default', 'selected': 5, 'predicted': 5, 'reviewed': 5,
            'sample_ids': ['0', '1', '2', '3', '4'], 'question_digest': 'qd',
            'media_digest': 'md', 'input_digest': 'id', 'media_evidence': True,
        }],
        'diagnostics': {'generation_calls': 5}, 'perf_metrics': None,
        'protocol_identity': 'pid', 'sample_manifest_identity': 'smid',
    }


def test_start_and_finish_roundtrip(tmp_path):
    store = Store(tmp_path / 'results.db')
    store.start_attempt(attempt_record())
    store.finish_attempt('run-1', outcome(0.0))
    row = store.get_attempt('run-1')
    assert row['execution_status'] == 'completed'
    assert row['validity_status'] == 'complete'
    assert row['comparability'] == 'verified'
    metric = store.conn.execute('SELECT * FROM metrics WHERE run_id=?', ('run-1',)).fetchone()
    assert metric['score'] == 0.0  # real zero is preserved
    manifest = store.latest_extra('run-1')
    assert manifest[0]['selected'] == 5
    assert store.conn.execute('PRAGMA journal_mode').fetchone()[0].lower() == 'wal'
    store.close()


def test_missing_scores_stay_null(tmp_path):
    store = Store(tmp_path / 'results.db')
    store.start_attempt(attempt_record())
    payload = outcome()
    payload['metrics'][0]['score'] = None
    payload['metrics'][0]['macro_score'] = None
    store.finish_attempt('run-1', payload)
    metric = store.conn.execute('SELECT score, macro_score FROM metrics').fetchone()
    assert metric['score'] is None and metric['macro_score'] is None
    store.close()


def test_duplicate_attempt_raises(tmp_path):
    store = Store(tmp_path / 'results.db')
    store.start_attempt(attempt_record())
    with pytest.raises(DuplicateAttemptError):
        store.start_attempt(attempt_record())
    store.close()


def test_finish_replaces_metrics_atomically(tmp_path):
    store = Store(tmp_path / 'results.db')
    store.start_attempt(attempt_record())
    store.finish_attempt('run-1', outcome(0.5))
    first = store.conn.execute('SELECT COUNT(*) AS n FROM metrics').fetchone()['n']
    payload = outcome(0.9)
    payload['metrics'].append({**payload['metrics'][0], 'metric_key': 'mk2', 'metric_name': 'f1'})
    store.finish_attempt('run-1', payload)
    rows = store.conn.execute('SELECT metric_key, score FROM metrics ORDER BY metric_key').fetchall()
    assert first == 1
    assert [(row['metric_key'], row['score']) for row in rows] == [('mk', 0.9), ('mk2', 0.9)]
    store.close()


def test_migration_from_v1_preserves_legacy_evidence(tmp_path):
    db = tmp_path / 'results.db'
    conn = sqlite3.connect(db)
    conn.executescript(V1_SCHEMA)
    conn.execute(
        """INSERT INTO runs VALUES ('v1-run', 'g', 'minimax', 'knowledge', 'smoke',
           'gpqa_diamond', 'success', 't1', 't2', 5, 5, 0, 0, '1.12.0', 'abc',
           '/tmp/out', '{}', NULL, NULL)"""
    )
    conn.execute("INSERT INTO metrics VALUES ('v1-run', 'gpqa_diamond', 'accuracy', 'default', 'default', 5, 0.8, 0.0)")
    conn.execute('INSERT INTO run_samples VALUES (?, ?, ?)', ('v1-run', 'gpqa_diamond', '["0", "1"]'))
    conn.commit()
    conn.close()

    store = Store(db)
    row = store.get_attempt('v1-run')
    assert row['validity_status'] == 'legacy'
    assert row['comparability'] == 'legacy'
    metric = store.conn.execute('SELECT * FROM metrics').fetchone()
    assert metric['metric_key'] == 'legacy:accuracy'
    assert metric['score'] == 0.8
    manifest = store.latest_extra('v1-run')
    assert manifest[0]['subset'] == 'legacy'
    assert manifest[0]['selected'] == 2
    assert store.conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0] == '2'
    store.close()


def test_migration_is_idempotent(tmp_path):
    db = tmp_path / 'results.db'
    Store(db).close()
    Store(db).close()  # second open must not fail or duplicate anything


def test_concurrent_writers_do_not_lock(tmp_path):
    db = tmp_path / 'results.db'
    first = Store(db)
    first.start_attempt(attempt_record('run-a'))
    errors = []

    def writer(run_id):
        try:
            store = Store(db)
            store.start_attempt(attempt_record(run_id))
            store.finish_attempt(run_id, outcome())
            store.close()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(f'run-b{i}',)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert first.conn.execute('SELECT COUNT(*) AS n FROM runs').fetchone()['n'] == 3
    first.close()
