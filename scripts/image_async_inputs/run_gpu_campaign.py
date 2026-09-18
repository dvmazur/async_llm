"""Durable GPU-replica orchestration for the suffix-only image evaluator."""
from __future__ import annotations

import concurrent.futures
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
from unittest.mock import patch

import image_async_thoughts_eval as ev
from fixed_subset import apply_manifest, select_ids
from report_fixed50 import report as fixed_report

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DATASET = REPO / 'diverse_corrections_v3.parquet'
MANIFEST = HERE / 'fixed_subset_50.json'
MODEL = '/mnt/LLM/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
SHA = '18125e129d92e04f459a9ee02cb6282c3372e3953a213c983026c7dadbe82a35'
GPUS = (1, 2, 3, 4, 6)
KS = (-1, 0, 16, 32, 64, 128, 256, 512)
CONTROL = HERE / 'eval_runs/qwen38_27b_suffix_v2_5gpu_16k'
FIXED = HERE / 'eval_runs/qwen38_27b_suffix_v2_5gpu_fixed50_16k'
FULL = HERE / 'eval_runs/qwen38_27b_suffix_v2_5gpu_full_16k'
PREFLIGHT = HERE / 'eval_runs/qwen38_27b_suffix_v2_5gpu_preflight_16k'
STOP = threading.Event()
MUTEX = threading.Lock()
CHILDREN = {}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(ok, message):
    if not ok:
        raise ValueError(message)


@contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def arguments(root, k, gpu):
    args = ['--dataset', str(DATASET), '--model-name', MODEL,
            '--output', str(root / f'k_{k}'), '--k-steps', str(k),
            '--budget', '16384', '--max-seq-len', '65536', '--kv-tokens', '98304',
            '--max-prefill-rows', '256', '--max-image-pixels', '1048576',
            '--distributed-port', str(24700 + gpu), '--shard-to-prompt',
            '--shard-to-thinker', '--writer-reminder', '--defer-writer-reminder']
    if root == FIXED:
        args += ['--sample-manifest', str(MANIFEST)]
    return args


def expected_config(root, k, gpu):
    # Parse with the evaluator itself so unspecified defaults remain identical.
    with patch.object(sys, 'argv', ['image_async_thoughts_eval.py', *arguments(root, k, gpu)]):
        a = ev.arguments()
    value = {key: str(v) if isinstance(v, Path) else v for key, v in vars(a).items()}
    value.update(dataset_sha256=SHA, sample_manifest=read(MANIFEST) if root == FIXED else None,
                 evaluator_sha256=digest(HERE / 'image_async_thoughts_eval.py'),
                 protocol=ev.PROTOCOL, hf_home='/mnt/LLM',
                 cuda_visible_devices=str(gpu))
    return value


def validate_condition(root, k, rows, indices, expected, *, complete, rescore=False):
    folder = root / f'k_{k}'
    if not (folder / 'config.json').exists():
        require(not list(folder.glob('*.json')), f'Unconfigured results: {folder}')
        require(not complete, f'Missing configuration: {folder}')
        return None
    require(read(folder / 'config.json') == expected, f'Exact config mismatch: {folder}')
    errors = list(folder.glob('*.error.json'))
    require(not errors, f'Saved sample errors require investigation: {errors}')
    ids = {r['id'] for r in rows}
    actual = {p.stem for p in folder.glob('*.json')} - {'config', 'summary'}
    require(actual <= ids, f'Unexpected result files: {folder}')
    results = []
    for position, row in enumerate(rows):
        path = folder / (row['id'] + '.json')
        if not path.exists():
            continue
        r = read(path)
        idx = indices[row['id']]
        require((r['id'], r['idx'], r['subset_position'], r['sample_seed']) ==
                (row['id'], idx, position, 42 + idx), f'Identity/order/seed mismatch: {path}')
        for field in ('dataset', 'category', 'difficulty_group'):
            require(r[field] == row[field], f'Metadata mismatch: {path}/{field}')
        for field in ('answer_before', 'answer_after'):
            require(r[field] == row['labels'][field], f'Reference answer mismatch: {path}')
        require(r['predicted_answer'] == ev.boxed(r['generated_text']), f'Box mismatch: {path}')
        nt, nw = [len(r['token_ids'][role]) for role in ('thinker', 'writer')]
        require(0 < nt <= 16384 and 0 < nw <= 16384, f'Invalid token budgets: {path}')
        require(r['hit_eos'] or nw == 16384, f'Unexplained termination: {path}')
        require(r['elapsed_seconds'] > 0 and r['image_tokens'] > 0, f'Invalid timing/image: {path}')
        probes = [e for e in r['events'] if e['event'] == 'probe']
        stats = r['probe_stats']
        require(stats['calls'] == len(probes) and stats['suffix_tokens'] > 0,
                f'Invalid probe count/suffix: {path}')
        require(0 <= stats['pending_tokens'] <= 2 * stats['calls'] and
                stats['input_tokens'] == stats['calls'] * stats['suffix_tokens'] + stats['pending_tokens'],
                f'Probe replay or invalid input-token accounting: {path}')
        require(0 <= stats['elapsed_seconds'] <= r['elapsed_seconds'], f'Invalid probe timing: {path}')
        replacements = [e for e in r['events'] if e['event'] == 'image_replaced']
        require(r['image_replaced'] == bool(replacements), f'Replacement flag mismatch: {path}')
        require(len(replacements) <= 1, f'Duplicate replacement: {path}')
        if k <= 0:
            require(not replacements, f'Baseline replaced its image: {path}')
        if replacements:
            require(k > 0 and replacements[0]['thinker_tokens'] == k, f'Wrong replacement timing: {path}')
        elif k > 0:
            require(nt <= k and r['hit_eos'], f'Missing required replacement: {path}')
        final_image = 'after' if k == 0 or replacements else 'before'
        require(r['final_image'] == final_image, f'Wrong final image: {path}')
        if rescore:
            for field, answer in [('correct_after', 'answer_after'), ('matches_before', 'answer_before')]:
                require(r[field] == ev.equivalent(r['predicted_answer'], row['labels'][answer],
                                                row['labels']['choices']), f'Score mismatch: {path}')
        results.append(r)
    if complete:
        require(len(results) == len(rows), f'Incomplete condition: {folder}: {len(results)}/{len(rows)}')
        require(read(folder / 'summary.json') == ev.summarize(results, len(rows)),
                f'Summary differs from sample results: {folder}')
    return results


def metrics(results, k):
    n = len(results)
    return {'completed': n, 'requested': n, 'errors': 0,
            'correct_after': sum(r['correct_after'] for r in results),
            'accuracy_after': sum(r['correct_after'] for r in results) / n,
            'matches_before': sum(r['matches_before'] for r in results),
            'image_replacements': sum(r['image_replaced'] for r in results),
            'missing_boxed_answers': sum(r['predicted_answer'] is None for r in results),
            'elapsed_seconds': sum(r['elapsed_seconds'] for r in results),
            'writer_eos': sum(r['hit_eos'] for r in results),
            'writer_budget_reached': sum(len(r['token_ids']['writer']) == 16384 for r in results),
            'thinker_budget_reached': sum(len(r['token_ids']['thinker']) == 16384 for r in results),
            'writer_ended_before_update': sum(k > 0 and not r['image_replaced'] for r in results),
            'deferred_reminder_pending': sum(r['writer_reminder_pending'] for r in results),
            'probe_calls': sum(r['probe_stats']['calls'] for r in results),
            'probe_input_tokens': sum(r['probe_stats']['input_tokens'] for r in results),
            'probe_elapsed_seconds': sum(r['probe_stats']['elapsed_seconds'] for r in results)}


def report(root, rows, indices, configs):
    conditions = []
    for k in KS:
        results = validate_condition(root, k, rows, indices, configs[k], complete=True, rescore=True)
        conditions.append({'k': k, **metrics(results, k), 'by_source': {
            source: metrics([r for r in results if r['dataset'] == source], k)
            for source in sorted({r['dataset'] for r in results})}})
    if root == FIXED:
        fixed_report(root, MANIFEST)
    value = {'complete': True, 'completed': len(rows) * 8, 'requested': len(rows) * 8,
             'errors': 0, 'dataset_sha256': SHA, 'conditions': conditions,
             'interpretation': 'k=-1 is the no-update original-image baseline, not a recovery run; '
                               'k=0 starts corrected. Positive k can finish before replacement.'}
    ev.save(root / 'validated_results.json', value)
    lines = [f'# {root.name}\n\n', value['interpretation'] + '\n\n',
             f"Completed/requested: {value['completed']}/{value['requested']}; errors: 0.\n\n",
             '| k | Completed | Corrected accuracy | Original matches | Replacements | Missing boxes | Seconds | Writer EOS | Writer cap | Thinker cap | Ended before update | Reminder pending |\n',
             '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n']
    for r in conditions:
        lines.append(f"| {r['k']} | {r['completed']}/{r['requested']} | {r['accuracy_after']:.2%} | "
                     f"{r['matches_before']} | {r['image_replacements']} | {r['missing_boxed_answers']} | "
                     f"{r['elapsed_seconds']:.1f} | {r['writer_eos']} | {r['writer_budget_reached']} | "
                     f"{r['thinker_budget_reached']} | {r['writer_ended_before_update']} | {r['deferred_reminder_pending']} |\n")
    lines += ['\n## Corrected-answer accuracy by source\n\n',
              '| Source | ' + ' | '.join(map(str, KS)) + ' |\n', '|---|' + '---:|' * 8 + '\n']
    for source in conditions[0]['by_source']:
        cells = [r['by_source'][source] for r in conditions]
        lines.append('| ' + source + ' | ' + ' | '.join(
            f"{r['correct_after']}/{r['completed']} ({r['accuracy_after']:.1%})" for r in cells) + ' |\n')
    lines += ['\n## Suffix-only probe workload\n\n',
              'Every probe was validated to process its fixed suffix and at most two pending tokens. '
              'Cached thinker/writer history is not re-prefilled. Protocol v2 results must not be pooled with v1.\n\n',
              '| k | Probe calls | Probe input tokens | Probe seconds |\n', '|---:|---:|---:|---:|\n']
    for r in conditions:
        lines.append(f"| {r['k']} | {r['probe_calls']} | {r['probe_input_tokens']} | {r['probe_elapsed_seconds']:.1f} |\n")
    (root / 'VALIDATED_REPORT.md').write_text(''.join(lines))
    print(f'VALIDATED {root}: {len(rows)*8}/{len(rows)*8}', flush=True)


def check_gpu(gpu):
    free = int(subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
    require(free >= 68000, f'GPU {gpu} has only {free} MiB free; require 68000 MiB')
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('127.0.0.1', 24700 + gpu))
    return free


def stop_children():
    STOP.set()
    with MUTEX:
        for child in list(CHILDREN.values()):
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass


def stage(root, rows, indices):
    root.mkdir(parents=True, exist_ok=True)
    pending = []
    configs = {}
    # Build configs serially: argument parsing temporarily changes sys.argv.
    candidates = {(k, gpu): expected_config(root, k, gpu) for k in KS for gpu in GPUS}
    for k in KS:
        folder = root / f'k_{k}'
        affinity = None
        if (folder / 'config.json').exists():
            old = read(folder / 'config.json')
            affinity = int(old['cuda_visible_devices'])
            require(affinity in GPUS, f'Saved condition pinned outside selected GPUs: {folder}')
            configs[k] = candidates[k, affinity]
            validate_condition(root, k, rows, indices, configs[k], complete=False)
            if (folder / 'summary.json').exists() and read(folder / 'summary.json').get('complete'):
                validate_condition(root, k, rows, indices, configs[k], complete=True)
                print(f'Skipping verified completed {root.name} k={k}', flush=True)
                continue
        pending.append((k, affinity))

    def worker(gpu):
        try:
            with lock(Path(f'/tmp/image_async_eval_gpu_{gpu}.lock')):
                while not STOP.is_set():
                    with MUTEX:
                        chosen = next((i for i, (_, pin) in enumerate(pending) if pin in (None, gpu)), None)
                        if chosen is None:
                            return
                        k, _ = pending.pop(chosen)
                        configs[k] = candidates[k, gpu]
                    folder = root / f'k_{k}'
                    with lock(folder / 'run.lock'):
                        validate_condition(root, k, rows, indices, configs[k], complete=False)
                        free = check_gpu(gpu)
                        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
                        command = ['uv', 'run', '--no-sync', 'python', '-u',
                                   str(HERE / 'image_async_thoughts_eval.py'), *arguments(root, k, gpu)]
                        with (root / f'k_{k}.gpu{gpu}.log').open('a', buffering=1) as log:
                            log.write(f'\nSTART {time.ctime()} GPU={gpu} free_MiB={free}\n')
                            with MUTEX:
                                if STOP.is_set():
                                    return
                                child = subprocess.Popen(command, cwd=REPO, env=env, stdout=log,
                                                         stderr=subprocess.STDOUT, start_new_session=True)
                                CHILDREN[gpu] = child
                                ev.save(CONTROL / f'gpu_{gpu}.json', {'gpu': gpu, 'k': k, 'pid': child.pid,
                                        'stage': root.name, 'output': str(folder), 'port': 24700+gpu,
                                        'status': 'running', 'started': time.time()})
                            print(f'START {root.name} GPU={gpu} k={k} PID={child.pid}', flush=True)
                            code = child.wait()
                            with MUTEX:
                                CHILDREN.pop(gpu, None)
                            require(code == 0, f'{root.name} GPU={gpu} k={k} exited {code}; see {log.name}')
                        validate_condition(root, k, rows, indices, configs[k], complete=True)
                        ev.save(CONTROL / f'gpu_{gpu}.json', {'gpu': gpu, 'k': k, 'status': 'complete',
                                                           'stage': root.name, 'finished': time.time()})
                        print(f'COMPLETE {root.name} GPU={gpu} k={k}', flush=True)
        except BaseException:
            stop_children()
            raise

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(GPUS)) as pool:
        futures = [pool.submit(worker, gpu) for gpu in GPUS]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    require(not pending and not STOP.is_set(), 'Stage stopped before all work succeeded')
    report(root, rows, indices, configs)


def main():
    os.chdir(REPO)
    require(os.environ.get('HF_HOME') == '/mnt/LLM', 'HF_HOME must be /mnt/LLM')
    CONTROL.mkdir(parents=True, exist_ok=True)
    with lock(CONTROL / 'campaign.lock'):
        require(digest(DATASET) == SHA, 'Dataset checksum mismatch')
        rows = ev.validate_package(DATASET)
        require(len(rows) == 513, 'Expected 513 rows')
        indices = {r['id']: i for i, r in enumerate(rows)}
        manifest = read(MANIFEST)
        require(manifest['sample_ids'] == select_ids(rows, 50, 42), 'Fixed manifest/order mismatch')
        fixed = apply_manifest(rows, manifest, SHA)
        require(read(PREFLIGHT / 'preflight.json') == {'count': 513, 'processor_parity': True},
                'All-513 processor preflight required')
        preflight = read(PREFLIGHT / 'config.json')
        require(preflight['evaluator_sha256'] == digest(HERE / 'image_async_thoughts_eval.py') and
                preflight['dataset_sha256'] == SHA and preflight['model_name'] == MODEL and
                preflight['max_image_pixels'] == 1048576, 'Stale processor preflight')
        files = subprocess.check_output(['git', 'ls-files', 'python', 'scripts/async_thoughts',
                                         'scripts/image_async_inputs', 'uv.lock'], text=True).splitlines()
        provenance = {'files': {f: digest(REPO / f) for f in files if not '/dataset_construction/' in f},
                      'orchestrator_sha256': digest(__file__), 'manifest_sha256': digest(MANIFEST),
                      'packages': {p: importlib.metadata.version(p) for p in
                                   ('torch', 'transformers', 'math-verify', 'latex2sympy2-extended', 'pyarrow')},
                      'gpus': GPUS, 'dataset_sha256': SHA, 'model': MODEL}
        provenance = json.loads(json.dumps(provenance))
        if (CONTROL / 'provenance.json').exists():
            require(read(CONTROL / 'provenance.json') == provenance, 'Campaign source/environment changed')
        ev.save(CONTROL / 'provenance.json', provenance)
        for gpu in GPUS:
            print(f'GPU {gpu}: {check_gpu(gpu)} MiB free', flush=True)
        try:
            stage(FIXED, fixed, indices)
            print('Fixed-50 validation passed; starting full 513-sample sweep', flush=True)
            stage(FULL, rows, indices)
            ev.save(CONTROL / 'complete.json', {'completed': 4504, 'fixed50': 400, 'full': 4104})
        except BaseException as exc:
            stop_children()
            ev.save(CONTROL / 'failure.json', {'error': repr(exc), 'time': time.time(),
                                             'full_launch_requires_fixed50_validation': True})
            raise


if __name__ == '__main__':
    main()
