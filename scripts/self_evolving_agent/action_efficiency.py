"""Actions per forward API call, including prefills and single-token decodes."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import time
from run_budget_sweep import retry_disk_full


class ForwardCounter:
    def __init__(self, llm):
        self.calls = 0
        llm.forward_counter = self
        original = llm.forward
        async def counted(*args, **kwargs):
            self.calls += 1
            return await original(*args, **kwargs)
        llm.forward = counted


def ratio(actions, forwards):
    return actions / forwards if forwards else None


def summarize(condition, task, runs):
    from scipy.stats import t
    values = [ratio(a, f) for a, f in runs]
    values = [v for v in values if v is not None]
    n = len(values)
    mean = statistics.mean(values) if n else None
    half = float(t.ppf(.975, n-1)) * statistics.stdev(values) / math.sqrt(n) if n > 1 else None
    forward_ratios = [f / a for a, f in runs if a]
    nf = len(forward_ratios)
    forward_mean = statistics.mean(forward_ratios) if nf else None
    forward_half = (float(t.ppf(.975, nf-1)) * statistics.stdev(forward_ratios) / math.sqrt(nf)
                    if nf > 1 else None)
    return dict(condition=condition, task=task, completed_runs=len(runs), ratio_runs=n,
                env_actions=sum(a for a, f in runs), forward_calls=sum(f for a, f in runs),
                mean_actions_per_forward=mean, ci95_half_width=half,
                forward_ratio_runs=nf, mean_forwards_per_env_step=forward_mean,
                forwards_per_env_step_ci95_half_width=forward_half)


@retry_disk_full
def report(root):
    state_path = root / "status.json"
    variant = json.loads(state_path.read_text()).get("prompt_variant", "minimal") if state_path.exists() else "minimal"
    rows = []
    for task in ('doom', 'health_gathering'):
        by_round = {i: [] for i in range(1, 6)}
        for run in range(10):
            out = root / variant / task / f'run{run:02}'
            if not (out / 'round_metrics.csv').exists() or not (out / 'task_results.log').exists():
                continue
            results = {}
            for line in (out / 'task_results.log').read_text().splitlines():
                try:
                    record = json.loads(line)
                    results[record['step']] = record
                except (ValueError, KeyError):
                    continue  # A concurrently written final line may be incomplete.
            with (out / 'round_metrics.csv').open() as f:
                for row in csv.DictReader(f):
                    if not row.get('valid_round') or row.get('invalid_reason'):
                        continue
                    result = results.get(int(row['round']))
                    if result and len(result['episodes']) == 5 and 'llm_forward_calls' in result:
                        by_round[int(row['valid_round'])].append((
                            sum(e['steps'] for e in result['episodes']), result['llm_forward_calls']))
        rows.extend(summarize(f'{variant}_round_{rnd}', task, runs) for rnd, runs in by_round.items())
    for name in ('reasoning', 'logit', 'no_think', 'random'):
        out = root / name
        if not (out / 'config.json').exists():
            continue
        cfg = json.loads((out / 'config.json').read_text())
        records = [json.loads(p.read_text()) for p in out.glob('episode_*.json')]
        for task in cfg['tasks']:
            for budget in cfg['budgets']:
                runs = []
                for run in range(cfg['runs']):
                    eps = [e for e in records if e['task']==task and e['budget']==budget and e['run']==run]
                    if len(eps)==cfg['episodes'] and all(not e['error'] and 'llm_forward_calls' in e for e in eps):
                        runs.append((sum(e['steps'] for e in eps), sum(e['llm_forward_calls'] for e in eps)))
                rows.append(summarize(f'{name}_{budget}', task, runs))
    tmp = root / 'action_efficiency.csv.tmp'
    with tmp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    tmp.replace(root / 'action_efficiency.csv')
    lines = ['# Environment actions per LLM forward', '',
             'Each run ratio = total completed environment actions / total LLM.forward() calls during its five episodes. '
             'Both prefill and decode calls count, including calls in decisions cancelled at episode end. '
             'Evolution/code generation and warm-up are excluded. These are API calls, not tokens or batched GPU launches. '
             'Report mean run ratio ± 95% Student-t CI across independent runs. Zero forwards gives N/A (random policy), not zero efficiency.', '',
             '| Condition | Env | Runs | Actions | Forwards | Actions/forward ± 95% CI |',
             '|---|---|---:|---:|---:|---:|']
    for r in rows:
        mean = 'N/A' if r['mean_actions_per_forward'] is None else f"{r['mean_actions_per_forward']:.5f}"
        half = '—' if r['ci95_half_width'] is None else f"{r['ci95_half_width']:.5f}"
        lines.append(f"| {r['condition']} | {r['task']} | {r['completed_runs']}/10 | {r['env_actions']} | {r['forward_calls']} | {mean} ± {half} |")
    tmp = root / 'action_efficiency.md.tmp'
    tmp.write_text('\n'.join(lines)+'\n'); tmp.replace(root / 'action_efficiency.md')
    interactivity = [dict(condition=r['condition'], task=r['task'], completed_runs=r['completed_runs'],
                         ratio_runs=r['forward_ratio_runs'], env_steps=r['env_actions'],
                         forward_calls=r['forward_calls'],
                         mean_forwards_per_env_step=r['mean_forwards_per_env_step'],
                         ci95_half_width=r['forwards_per_env_step_ci95_half_width']) for r in rows]
    tmp = root / 'interactivity.csv.tmp'
    with tmp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(interactivity[0]))
        writer.writeheader(); writer.writerows(interactivity)
    tmp.replace(root / 'interactivity.csv')
    lines = ['# LLM forwards per environment step', '',
             'Each run ratio is total evaluation LLM.forward() calls / total completed environment steps '
             'over five episodes. Report the mean run ratio with 95% Student-t CI; do not invert the mean '
             'actions/forward ratio. Prefill and decode calls count; warm-up and code generation do not. '
             'Random is zero; runs with zero completed steps have undefined ratios and are excluded from the CI. '
             'Lower values mean fewer forwards per decision update; held game tics are not extra steps.', '',
             '| Condition | Env | Completed runs | Defined ratios | Steps | Forwards | Forwards/step ± 95% CI |',
             '|---|---|---:|---:|---:|---:|---:|']
    for r in interactivity:
        value = 'N/A' if r['mean_forwards_per_env_step'] is None else f"{r['mean_forwards_per_env_step']:.4f}"
        half = '—' if r['ci95_half_width'] is None else f"{r['ci95_half_width']:.4f}"
        lines.append(f"| {r['condition']} | {r['task']} | {r['completed_runs']}/10 | {r['ratio_runs']} | "
                     f"{r['env_steps']} | {r['forward_calls']} | {value} ± {half} |")
    tmp = root / 'interactivity.md.tmp'
    tmp.write_text('\n'.join(lines)+'\n'); tmp.replace(root / 'interactivity.md')
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--watch', action='store_true')
    args=p.parse_args()
    while True:
        report(args.root)
        status=json.loads((args.root/'status.json').read_text())
        if not args.watch or status['status'] in ('complete','failed','stopped'):
            break
        time.sleep(30)
