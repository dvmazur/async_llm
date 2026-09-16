"""CLI usable directly from an unpacked source tree, without entry-point installs."""
import sys


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ('summary', 'pack'):
        raise SystemExit('usage: python -m experiment_runner {summary|pack} ...; setup: python -m environment.setup')
    command = sys.argv.pop(1)
    if command == 'summary':
        from .summary import main as run
    else:
        from .artifacts import main as run
    run()


if __name__ == '__main__':
    main()
