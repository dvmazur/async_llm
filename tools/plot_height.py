"""Local-only plotting (python -m tools.plot_height). No simulator or CUDA imports."""
import argparse
import json
from pathlib import Path


def aggregate(episodes, *, descent=False):
    points = {}
    for episode in episodes:
        if episode['status'] != 'completed':
            continue
        trajectory = episode['heights']
        initial = next((r['height'] for r in trajectory if r['step'] == 0), None)
        for row in trajectory:
            value = row['height']
            if value is None or (descent and initial is None):
                continue
            points.setdefault(row['step'], []).append(initial-value if descent else value)
    return [dict(step=i, mean=sum(v)/len(v), count=len(v)) for i, v in sorted(points.items())]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', action='append', required=True, metavar='LABEL=DIRECTORY')
    p.add_argument('--output', required=True)
    p.add_argument('--descent', action='store_true')
    args = p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    tables = {}
    for entry in args.run:
        label, path = entry.split('=', 1)
        data = json.loads((Path(path)/'analysis/summary.json').read_text())
        points = aggregate(data['episodes'], descent=args.descent)
        tables[label] = points
        ax.plot([r['step'] for r in points], [r['mean'] for r in points], label=label)
    ax.set(xlabel='Action', ylabel='Mean descent from start' if args.descent else 'Mean height')
    ax.legend()
    ax.grid(alpha=.2)
    fig.savefig(args.output, bbox_inches='tight')
    Path(args.output).with_suffix('.json').write_text(json.dumps(tables, indent=2))


if __name__ == '__main__':
    main()
