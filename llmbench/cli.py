"""Command line interface: run / prepare / report / import."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__ as TOOL_VERSION  # noqa: F401  (kept for --version consumers)
from .config import (
    PROFILE_NAMES,
    SUITE_NAMES,
    ConfigError,
    build_plan,
    load_models,
    load_suites,
    module_available,
    preflight,
    validate_pinned_dir,
)
from .images import cleanup_owned_images, collect_owned_images
from .prepare import prepare_django_dataset
from .runner import (
    BatchInterrupted,
    batch_exit_code,
    build_raw_task_config,
    import_output,
    resolve_task_config,
    run_plan,
)
from .store import Store
from .summary import write as write_summary

REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(os.environ.get('LLMBENCH_DATA_ROOT', '/ssd4/LLMBenchmark'))


def setup_cache_env(data_root: Path) -> None:
    os.environ.setdefault('EVALSCOPE_CACHE', str(data_root / 'evalscope-cache'))
    os.environ.setdefault('HF_HOME', str(data_root / 'hf-cache'))
    os.environ.setdefault('HF_DATASETS_CACHE', str(data_root / 'hf-cache' / 'datasets'))


def _store(args) -> Store:
    db_path = Path(args.db) if getattr(args, 'db', None) else Path(args.data_root).resolve() / 'results.db'
    return Store(db_path)


def cmd_run(args) -> int:
    repo_dir = REPO_DIR
    data_root = Path(args.data_root).resolve()
    try:
        models_cfg = load_models(Path(args.config_dir))
        suites_doc = load_suites(Path(args.config_dir))
        plan = build_plan(
            models_cfg=models_cfg, suites_doc=suites_doc, data_root=data_root,
            model_names=args.model or ['minimax'], suite_names=[args.suite] if args.suite != 'all'
            else list(SUITE_NAMES),
            profile=args.profile, datasets=args.datasets, limit=args.limit,
            batch_size=args.batch_size, include_extensions=args.include_extensions,
        )
    except ConfigError as exc:
        print(f'config error: {exc}', file=sys.stderr)
        return 2

    if args.dry_run:
        return _print_dry_run(plan, data_root)
    return _execute(plan, args, repo_dir, data_root)


def _print_dry_run(plan, data_root: Path) -> int:
    print(json.dumps({
        'data_root': str(data_root),
        'profile': plan.profile,
        'dry_run': True,
        'entries': [{
            'model_alias': entry.model_alias,
            'model_id': entry.model_cfg['model_id'],
            'suite': entry.suite,
            'dataset': entry.dataset,
            'per_subset_limit': entry.limit,
            'batch_size': entry.batch_size,
            'pinned': entry.pinned,
            'local_path': str(entry.local_path) if entry.local_path else None,
            'pinned_ready': (validate_pinned_dir(entry.local_path)[0]
                             if entry.pinned and entry.local_path else None),
            'requires': list(entry.requires),
            'requires_missing': [m for m in entry.requires if not module_available(m)],
            'status': entry.status,
            'output_base': str(data_root / 'outputs'),
        } for entry in plan.entries],
    }, ensure_ascii=False, indent=2))
    return 0


def _preflight_errors(plan, data_root: Path) -> list:
    errors = list(preflight(plan))
    for entry in plan.entries:
        try:
            raw_cfg = build_raw_task_config(entry, data_root=data_root,
                                            output_dir=data_root / 'outputs' / 'preflight',
                                            api_key='EMPTY')
            resolve_task_config(raw_cfg)
        except Exception as exc:  # noqa: BLE001 - report all config errors before spending money
            errors.append(f'{entry.dataset}: invalid task config ({exc.__class__.__name__}: {exc})')
    return errors


def _execute(plan, args, repo_dir: Path, data_root: Path) -> int:
    setup_cache_env(data_root)
    errors = _preflight_errors(plan, data_root)
    if errors:
        print('preflight failed:', file=sys.stderr)
        for error in errors:
            print(f'  - {error}', file=sys.stderr)
        return 2

    try:
        store = _store(args)
    except Exception as exc:  # noqa: BLE001
        print(f'cannot open results database: {exc}', file=sys.stderr)
        return 2

    results = []
    interrupted = False
    try:
        try:
            results = run_plan(plan, store=store, repo_dir=repo_dir)
        except BatchInterrupted:
            interrupted = True
            print('interrupted: remaining tasks were not started', file=sys.stderr)
        except KeyboardInterrupt:
            interrupted = True

        if args.cleanup_images:
            images = collect_owned_images([result.output_dir for result in results])
            if images:
                report = cleanup_owned_images(images)
                print(f"image cleanup: removed={report['removed']} kept={report['kept']}")
            else:
                print('image cleanup: no owned SWE-bench images detected; nothing removed')

        summary_path = write_summary(
            store, repo_dir / 'results' / 'summary.md', data_root,
            include_partial=args.allow_partial,
        )
        print(f'summary written to {summary_path}')
    finally:
        store.close()

    if interrupted:
        return 130
    return batch_exit_code(results, allow_partial=args.allow_partial)


def cmd_prepare(args) -> int:
    data_root = Path(args.data_root).resolve()
    setup_cache_env(data_root)
    try:
        result = prepare_django_dataset(data_root, force=args.force)
    except Exception as exc:  # noqa: BLE001
        print(f'prepare failed: {exc}', file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result['status'] in ('created', 'exists') else 1


def cmd_report(args) -> int:
    data_root = Path(args.data_root).resolve()
    store = _store(args)
    try:
        out = write_summary(store, Path(args.out) if args.out else REPO_DIR / 'results' / 'summary.md',
                            data_root, include_partial=args.allow_partial)
    finally:
        store.close()
    print(f'wrote {out}')
    return 0


def cmd_import(args) -> int:
    repo_dir = REPO_DIR
    data_root = Path(args.data_root).resolve()
    setup_cache_env(data_root)
    store = _store(args)
    try:
        result = import_output(Path(args.output_dir), store, repo_dir=repo_dir, data_root=data_root)
    except Exception as exc:  # noqa: BLE001
        print(f'import failed: {exc.__class__.__name__}: {exc}', file=sys.stderr)
        print('existing scores were left untouched', file=sys.stderr)
        return 1
    finally:
        store.close()
    print(json.dumps(result, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    def add_common(p, db=True):
        p.add_argument('--data-root', default=str(DEFAULT_DATA_ROOT),
                       help=f'data/output root (default: {DEFAULT_DATA_ROOT})')
        p.add_argument('--config-dir', default=str(REPO_DIR / 'configs'))
        if db:
            p.add_argument('--db', default=None, help='SQLite path (default: <data-root>/results.db)')

    run_p = sub.add_parser('run', help='run benchmarks')
    add_common(run_p)
    run_p.add_argument('--model', action='append', default=None)
    run_p.add_argument('--suite', choices=(*SUITE_NAMES, 'all'), default='all')
    run_p.add_argument('--profile', choices=PROFILE_NAMES, default='lite')
    run_p.add_argument('--datasets', nargs='+', default=None,
                       help='explicit dataset selection; overrides the core filter')
    run_p.add_argument('--limit', type=int, default=None)
    run_p.add_argument('--batch-size', type=int, default=None)
    run_p.add_argument('--include-extensions', action='store_true')
    run_p.add_argument('--allow-partial', action='store_true',
                       help='treat partial runs as acceptable and show them in the formal table')
    run_p.add_argument('--cleanup-images', action='store_true',
                       help='remove only images this run provably owns and that are not in use')
    run_p.add_argument('--dry-run', action='store_true',
                       help='print the plan without keys, DB, Docker or downloads')

    prep_p = sub.add_parser('prepare', help='build the pinned SWE-bench Django dataset')
    add_common(prep_p, db=False)
    prep_p.add_argument('--force', action='store_true')

    rep_p = sub.add_parser('report', help='regenerate results/summary.md from the DB')
    add_common(rep_p)
    rep_p.add_argument('--out', default=None)
    rep_p.add_argument('--allow-partial', action='store_true')

    imp_p = sub.add_parser('import', help='re-parse an attempt output directory (no model calls)')
    add_common(imp_p)
    imp_p.add_argument('--output-dir', required=True)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == 'run':
        return cmd_run(args)
    if args.command == 'prepare':
        return cmd_prepare(args)
    if args.command == 'report':
        return cmd_report(args)
    if args.command == 'import':
        return cmd_import(args)
    parser.error(f'unknown command {args.command}')
