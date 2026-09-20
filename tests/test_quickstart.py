from pathlib import Path


def test_portable_15x10_matches_bench_and_disables_media():
    from experiments import speleo_15x10_sequential as run
    from experiments.speleo_15x5_sequential import ENGINE_PARAMS
    assert run.ENGINE_PARAMS == ENGINE_PARAMS
    assert (run.PIPELINES_PER_GPU, run.REPEATS, run.ACTIONS, run.GPUS) == (15, 1, 10, [0])
    assert run.DUMP_IMAGES is False and run.GIF_ON is False
    assert run.VENV == run.REPOSITORY / '.venvs' / 'minisgl'
    assert run.RESULTS == run.REPOSITORY / 'results' / 'speleo_15x10_sequential'
    assert run.MODEL_SEED_START == run.WORLD_SEED_START == 0


def test_release_has_readme_documentation_and_no_machine_specific_experiments():
    from tools.build_release import source_files
    root = Path(__file__).resolve().parents[1]
    paths = {str(p.relative_to(root)) for p in source_files(root)}
    assert {'README.md', 'DOCUMENTATION.md'} <= paths
    assert 'QUICKSTART.md' not in paths
    assert 'experiments/speleo_15x10_sequential.py' in paths
    assert 'experiments/sglang_15x150_r7_sequential.py' in paths
    assert 'experiments/minisgl_15x150_r7_sequential.py' in paths
    assert 'environment/sglang/uv.lock' in paths
    assert not any(p.startswith('experiments/rtx_') for p in paths)


def test_current_sglang_evaluation_and_pipeline_factory(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from experiments import sglang_15x150_r7_sequential as run
    assert (run.PIPELINES_PER_GPU, run.ACTIONS, run.REPEATS, run.GPUS) == (15, 150, 7, [0])
    assert run.ENGINE_PARAMS['backend'] == 'sglang'
    config = run.ENGINE_PARAMS['engine_config']
    assert config['model_path'] == str(run.MODEL)
    assert run.MODEL.name == 'Qwen3.6-35B-A3B-FP8'
    assert config['context_length'] == 131072
    assert config['max_running_requests'] >= run.PIPELINES_PER_GPU
    assert config['cuda_graph_max_bs_decode'] >= run.PIPELINES_PER_GPU
    assert run.VENV == run.REPOSITORY/'.venvs/sglang'
    assert run.RESULTS.name == 'sglang_15x150_r7_sequential'
    monkeypatch.setattr(run, 'SpeleoWorld', lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(run, 'Recorder', lambda directory, **kwargs: SimpleNamespace(directory=directory, **kwargs))
    context = SimpleNamespace(world_seed=42, results_directory=tmp_path)
    engine = object()
    pipeline = run.make_pipeline(engine, context)
    assert pipeline.engine is engine and pipeline.max_actions == 150
    assert pipeline.planner_interval == 10
    assert pipeline.world.max_steps == 150 and pipeline.world.seed == 42
    assert pipeline.world.craftium_directory == str(run.CRAFTIUM)
    assert pipeline.recorder.directory == tmp_path
    assert pipeline.recorder.dump_images is False and pipeline.recorder.gif_on is False


def test_long_minisgl_has_context_and_aggregate_kv_capacity():
    from experiments import minisgl_15x150_r7_sequential as run
    from experiments import speleo_15x10_sequential as short
    assert (run.PIPELINES_PER_GPU, run.ACTIONS, run.REPEATS) == (15, 150, 7)
    assert run.VENV == run.REPOSITORY/'.venvs/minisgl'
    assert run.ENGINE_PARAMS.get('backend', 'minisgl') == 'minisgl'
    config = run.ENGINE_PARAMS['engine_config']
    assert config['max_seq_len_override'] == 131072
    assert config['num_page_override'] * config['page_size'] == 1572864
    assert run.DUMP_IMAGES is False and run.GIF_ON is False
    # Keep measured short benchmarks reproducible. Only the two capacity
    # overrides change, not precision, graph catalogue, prefill caps or policy.
    changed = {key for key, value in config.items()
               if value != short.ENGINE_PARAMS['engine_config'][key]}
    assert changed == {'max_seq_len_override', 'num_page_override'}
    assert run.ROLE_PARAMETERS == short.ROLE_PARAMETERS
    assert run.RESULTS.name == 'minisgl_15x150_r7_sequential'


def test_source_release_needs_no_engine_checkout(tmp_path):
    import hashlib
    import json
    import zipfile
    from tools.build_release import build
    output = tmp_path/'runner.zip'
    build(output)
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        assert not any(name.endswith('.bundle') for name in archive.namelist())
        manifest = json.loads(archive.read('speleo-runner/tools/release.json'))
        assert manifest['includes_engine'] is False
        for name, expected in manifest['sha256'].items():
            assert hashlib.sha256(archive.read('speleo-runner/'+name)).hexdigest() == expected


def test_each_experiment_declares_its_own_parameters():
    import ast
    root = Path(__file__).resolve().parents[1]
    for path in (root/'experiments').glob('*.py'):
        if path.name == '__init__.py':
            continue
        assert path.stem.endswith('_sequential'), path
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or '').startswith('experiments'), path
                assert not any(alias.name == 'ROLE_PARAMS' for alias in node.names), path
        assignments = {target.id: node.value for node in tree.body if isinstance(node, ast.Assign)
                       for target in node.targets if isinstance(target, ast.Name)}
        params = assignments.get('ENGINE_PARAMS', assignments.get('PARAMS'))
        assert isinstance(params, ast.Dict), path
        roles = assignments['ROLE_PARAMETERS']
        assert isinstance(roles, ast.Dict), path
        assert {ast.literal_eval(key) for key in roles.keys} == {
            'observer', 'planner', 'executor'}, path


def test_historical_layouts_preserved_when_inlining():
    import ast
    root = Path(__file__).resolve().parents[1]/'experiments'
    expected = {
        'rtx_15x100_cuda_graphs_sequential': (24576, 1024, [4, 16, 48, 64], [256, 1024], 'fp8', False),
        'rtx_15x10_24d940f_sequential': (8192, 4096, [4, 16, 48, 64], [256, 1024, 4096], 'fp8', False),
        'rtx_15x10_cuda_graphs_sequential': (8192, 4096, [4, 16, 48, 64], [256, 1024, 4096], 'fp8', False),
        'rtx_15x10_gdn_bf16_sequential': (8192, 4096, [4, 16, 48, 64], [256, 1024, 4096], 'fp8', True),
        'rtx_bf16_1x10_prefill_graphs_sequential': (8192, 4096, [4], [256, 1024, 4096], None, False),
    }
    for name, layout in expected.items():
        tree = ast.parse((root/(name+'.py')).read_text())
        params = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'PARAMS' for t in n.targets))
        config = params['engine_config']
        assert (config['num_page_override'], config['max_prefill_rows'], config['cuda_graph_bs'],
                config['shared_cuda_graph_prefill_rows'], config.get('quantization'),
                config.get('shared_gdn_bf16_state', False)) == layout
        assert config['cuda_graph_max_bs'] == max(config['cuda_graph_bs'])
        assert config['dtype'] == 'bfloat16'
        assert config['shared_cuda_graph_max_depth'] == 16
        assert params['adapter_options'] == {'cpu_threads': 4}
