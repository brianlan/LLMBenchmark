"""Integration test against artifacts in the real EvalScope report/prediction format.

The report, reviews and predictions are verbatim copies of a real GPQA-Diamond
smoke run (see tests/fixtures/README.md); only the model call is replaced, so the
whole parse -> validate -> persist -> summary path runs on real data.
"""

import json
import shutil
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import llmbench.runner as runner
from llmbench.config import PlanEntry
from llmbench.store import Store
from llmbench.summary import render

FIXTURES = Path(__file__).resolve().parent / 'fixtures'
MODEL_ID = 'MiniMax-M3.1-Flash-Preview'
MODEL_CFG = {
    'model_id': MODEL_ID,
    'report_id': MODEL_ID,
    'api_url': 'https://api.minimax.cn/v1',
    'api_key': 'fixture-key',
}


@contextmanager
def fake_capture(callback):
    samples = [
        type('Sample', (), {'id': index, 'input': f'fixture question {index}'})()
        for index in range(5)
    ]
    callback(None, {'default': samples})
    yield {}


def copy_real_output(output_dir: Path, dataset='gpqa_diamond'):
    (output_dir / 'reports' / MODEL_ID).mkdir(parents=True, exist_ok=True)
    (output_dir / 'predictions' / MODEL_ID).mkdir(parents=True, exist_ok=True)
    (output_dir / 'reviews' / MODEL_ID).mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURES / 'report_gpqa_diamond.json',
                output_dir / 'reports' / MODEL_ID / f'{dataset}.json')
    shutil.copy(FIXTURES / 'predictions_gpqa_diamond_default.jsonl',
                output_dir / 'predictions' / MODEL_ID / f'{dataset}_default.jsonl')
    shutil.copy(FIXTURES / 'reviews_gpqa_diamond_default.jsonl',
                output_dir / 'reviews' / MODEL_ID / f'{dataset}_default.jsonl')


@contextmanager
def stub_ocr_patch():
    yield {'applied': True, 'reason': 'test', 'target': 'nltk.edit_distance'}


def test_real_eval_scope_format_end_to_end(tmp_path, monkeypatch, capsys):
    module = types.ModuleType('evalscope')
    module.__version__ = '1.12.0'

    def run_task(cfg):
        copy_real_output(Path(cfg['work_dir']))

    module.run_task = run_task
    monkeypatch.setitem(sys.modules, 'evalscope', module)
    monkeypatch.setattr(runner, 'capture_loaded_dataset', fake_capture)
    monkeypatch.setattr(runner, 'resolve_task_config', lambda raw: {'model': raw['model']})
    monkeypatch.setattr(runner, 'ocr_compat_patch', stub_ocr_patch)
    monkeypatch.setattr(runner, 'report_patch_status', lambda status: None)

    entry = PlanEntry(
        model_alias='minimax', model_cfg=MODEL_CFG, suite='knowledge',
        dataset='gpqa_diamond', spec={}, profile='lite', limit=5, batch_size=4,
        pinned=False, status='tested',
    )
    output_dir = tmp_path / 'outputs' / 'attempt'
    raw_cfg = {
        'model': MODEL_ID, 'model_id': MODEL_ID, 'api_url': MODEL_CFG['api_url'],
        'api_key': 'fixture-key', 'datasets': ['gpqa_diamond'],
        'work_dir': str(output_dir),
    }
    store = Store(tmp_path / 'results.db')
    result = runner.execute_attempt(
        entry, store=store, attempt_id='real-format-1', run_group='group',
        output_dir=output_dir, raw_cfg=raw_cfg, resolved_cfg={'model': MODEL_ID},
        repo_dir=tmp_path, data_root=tmp_path,
    )

    assert (result.execution_status, result.validity_status) == ('completed', 'complete')
    row = store.get_attempt('real-format-1')
    assert row['comparability'] == 'verified'
    assert row['num_requested'] == 5 and row['num_succeeded'] == 5
    primary = store.conn.execute(
        'SELECT * FROM metrics WHERE run_id=? AND is_primary=1', ('real-format-1',)
    ).fetchone()
    assert primary is not None
    assert primary['num'] == 5
    assert abs(primary['score'] - 0.8) < 1e-9  # the real recorded score
    manifest = store.latest_extra('real-format-1')[0]
    assert manifest['selected'] == 5 and manifest['predicted'] == 5 and manifest['reviewed'] == 5

    diagnostics = json.loads(row['diagnostics_json'])
    assert diagnostics['generation_calls'] == 5
    assert diagnostics['truncated_calls'] == 0
    assert diagnostics['samples_without_stop_reason'] == 0

    text = render(store, tmp_path)
    formal = text.split('## Diagnostics')[0]
    assert 'real-format-1' in formal
    assert '80.0%' in formal
    store.close()
