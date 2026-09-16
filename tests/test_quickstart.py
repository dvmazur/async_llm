from pathlib import Path


def test_portable_15x10_matches_bench_and_disables_media():
    from experiments import speleo_15x10 as run
    from experiments.speleo_15x5 import ENGINE_PARAMS
    assert run.ENGINE_PARAMS == ENGINE_PARAMS
    assert (run.PIPELINES_PER_GPU, run.REPEATS, run.ACTIONS, run.GPUS) == (15, 1, 10, [0])
    assert run.DUMP_IMAGES is False and run.GIF_ON is False
    assert run.VENV == run.REPOSITORY / '.venvs' / 'minisgl'
    assert run.RESULTS == run.REPOSITORY / 'results' / 'speleo-15x10'
    assert run.MODEL_SEED_START == run.WORLD_SEED_START == 0


def test_release_has_quickstart_and_no_machine_specific_experiments():
    from tools.build_release import source_files
    root = Path(__file__).resolve().parents[1]
    paths = {str(p.relative_to(root)) for p in source_files(root)}
    assert 'QUICKSTART.md' in paths
    assert 'experiments/speleo_15x10.py' in paths
    assert not any(p.startswith('experiments/rtx_') for p in paths)
