"""Equal TP selection, local checkpoint shapes and numerical parity."""

from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace

import minisgl.distributed.info as info
import pytest
import torch
import torch.nn.functional as F
from minisgl.distributed import DistributedInfo
from minisgl.distributed.partition import (
    select_tp_size,
    validate_model_partition,
)
from minisgl.kvcache import GDNStatePool, create_kvcache_pool
from minisgl.models.config import ModelConfig
from minisgl.models.qwen3_5_attn import Qwen3_5Attention
from minisgl.models.qwen3_5_delta import Qwen3_5GatedDeltaNet
from minisgl.models.utils import GatedMLP
from minisgl.models.weight import _get_merge_info, _shard_tensor
from transformers import Qwen3_5Config


def _config(*, large=False):
    dims = {
        "hidden_size": 32,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 8,
        "linear_num_key_heads": 4,
        "linear_num_value_heads": 12,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "intermediate_size": 48,
    }
    if large:
        dims.update(
            hidden_size=5120,
            num_attention_heads=24,
            num_key_value_heads=4,
            head_dim=256,
            linear_num_key_heads=16,
            linear_num_value_heads=48,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            intermediate_size=17408,
        )
    return ModelConfig.from_hf(
        Qwen3_5Config(
            text_config=dict(
                **dims,
                vocab_size=31,
                num_hidden_layers=2,
                layer_types=["linear_attention", "full_attention"],
                linear_conv_kernel_dim=4,
                attn_output_gate=True,
            )
        )
    )


@pytest.mark.parametrize(
    "requested,expected",
    [
        (1, 1),
        (2, 2),
        (3, 2),
        (4, 4),
        (5, 4),
        (6, 4),
        (7, 4),
        (8, 8),
        (12, 8),
        (16, 8),
        (24, 8),
        (32, 8),
    ],
)
def test_selects_largest_supported_tp(requested, expected):
    config = replace(_config(large=True), weight_block_size=128)
    assert select_tp_size(config, requested, available=32) == expected


def test_selection_respects_visible_devices_and_fp8_blocks():
    config = replace(_config(large=True), weight_block_size=128)
    assert select_tp_size(config, 8, available=3) == 2
    assert select_tp_size(config, 8, available=1) == 1
    # Head counts allow eight ranks, but ten FP8 MLP blocks only divide by two.
    config = replace(config, intermediate_size=1280)
    assert select_tp_size(config, 8, available=8) == 2
    assert select_tp_size(replace(config, weight_block_size=1), 8, available=8) == 8


def test_selection_is_not_limited_to_powers_of_two():
    config = replace(
        _config(large=True),
        num_qo_heads=12,
        num_kv_heads=6,
        linear_num_key_heads=6,
        linear_num_value_heads=12,
        intermediate_size=768,
        weight_block_size=128,
    )
    assert select_tp_size(config, 5, available=8) == 3
    assert select_tp_size(config, 8, available=8) == 6


@pytest.mark.parametrize("requested,available", [(0, 4), (-1, 4), (4, 0)])
def test_invalid_device_counts(requested, available):
    with pytest.raises(ValueError):
        select_tp_size(_config(), requested, available)


@pytest.mark.parametrize("size", [3, 6, 12, 16, 25])
def test_direct_engine_layout_remains_strict(size):
    with pytest.raises(ValueError, match="equal shards"):
        validate_model_partition(_config(large=True), size)


def _modules(config):
    return {
        "model.layers.0.self_attn": Qwen3_5Attention(config, 0),
        "model.layers.0.linear_attn": Qwen3_5GatedDeltaNet(config, 0),
        "model.layers.0.mlp": GatedMLP(config),
    }


def _checkpoint(modules, *, quantized=False):
    result = {}
    for prefix, module in modules.items():
        for key, weight in module.state_dict(prefix=prefix).items():
            parts = [(key, weight)]
            if ".qkv_proj.weight" in key:
                # The on-disk attention checkpoint has separate Q+gate, K, V.
                widths = [2 * module.qo_dim, module.kv_dim, module.kv_dim]
                parts = [
                    (key.replace("qkv_proj", name), value)
                    for name, value in zip(("q_proj", "k_proj", "v_proj"), weight.split(widths))
                ]
            elif ".gate_up_proj.weight" in key:
                parts = [
                    (key.replace("gate_up_proj", name), value)
                    for name, value in zip(("gate_proj", "up_proj"), weight.chunk(2))
                ]
            for name, value in parts:
                is_fp8 = quantized and value.ndim == 2 and all(d % 128 == 0 for d in value.shape)
                result[name] = value.to(torch.float8_e4m3fn) if is_fp8 else value
                if is_fp8:
                    result[name + "_scale_inv"] = torch.ones(
                        value.shape[0] // 128, value.shape[1] // 128, device=value.device
                    )
    return result


def _load(modules, checkpoint, config, rank, size):
    state, pending = {}, {}
    for key, value in checkpoint.items():
        value = _shard_tensor(key, value, rank, size, config)
        merge = _get_merge_info(key)
        if merge is None:
            state[key] = value
        else:
            name, slot, slots = merge
            pending.setdefault(name, {})[slot] = value
            if all(s in pending[name] for s in slots):
                state[name] = torch.cat([pending[name][s] for s in slots], dim=0)
                del pending[name]
    assert not pending
    for prefix, module in modules.items():
        module.load_state_dict(state, prefix=prefix, _internal=True)
    assert not state


@pytest.mark.parametrize("size", [1, 2, 4, 8])
def test_real_27b_fp8_shapes_and_caches_for_every_rank(monkeypatch, size):
    config = replace(_config(large=True), weight_block_size=128)
    validate_model_partition(config, size)
    monkeypatch.setattr(info, "_TP_INFO", DistributedInfo(0, 1))
    with torch.device("meta"):
        checkpoint = _checkpoint(_modules(config), quantized=True)
        for rank in range(size):
            monkeypatch.setattr(info, "_TP_INFO", DistributedInfo(rank, size))
            modules = _modules(config)
            _load(modules, checkpoint, config, rank, size)
            attention, delta, mlp = modules.values()
            kv = create_kvcache_pool(config, 3, 1, torch.float32, torch.device("meta"))
            pool = GDNStatePool(config, 2, torch.device("meta"), torch.float32)
            assert kv.k_cache(0).shape[-2] == attention.num_kv_heads
            assert pool.conv_state.shape[-2] == delta.conv_dim
            assert pool.recurrent_state.shape[-3] == delta.num_v_heads
            assert mlp.gate_up_proj.local_output_size == 2 * mlp.down_proj.local_input_size


@pytest.mark.parametrize("size", [2, 4, 8])
def test_equal_fp8_scales_follow_the_original_blocks(size):
    config = replace(_config(large=True), weight_block_size=128)
    cases = {
        "self_attn.q_proj.weight": (96, 1),
        "self_attn.k_proj.weight": (8, 1),
        "self_attn.v_proj.weight": (8, 1),
        "self_attn.o_proj.weight": (1, 48),
        "linear_attn.in_proj_qkv.weight": (80, 1),
        "linear_attn.in_proj_z.weight": (48, 1),
        "linear_attn.out_proj.weight": (1, 48),
        "mlp.gate_proj.weight": (136, 1),
        "mlp.down_proj.weight": (1, 136),
    }
    for name, shape in cases.items():
        scales = torch.arange(shape[0] * shape[1], dtype=torch.float32).reshape(shape)
        weight = scales.repeat_interleave(128, 0).repeat_interleave(128, 1)
        for rank in range(size):
            key = "model.layers.0." + name
            w = _shard_tensor(key, weight, rank, size, config)
            s = _shard_tensor(key + "_scale_inv", scales, rank, size, config)
            torch.testing.assert_close(w[::128, ::128], s, rtol=0, atol=0)
            assert w.shape == (s.shape[0] * 128, s.shape[1] * 128)


def test_vocabulary_padding_and_batch_gather(monkeypatch):
    import minisgl.layers.embedding as embedding
    from minisgl.layers import ParallelLMHead

    weight, x = torch.randn(31, 8), torch.randn(2, 8)
    shards = [_shard_tensor("lm_head.weight", weight, r, 3, _config()) for r in range(3)]
    assert [s.shape[0] for s in shards] == [11, 11, 11]
    assert not shards[-1][-2:].count_nonzero()
    gathered = torch.cat([F.linear(x, shard) for shard in shards])
    monkeypatch.setattr(info, "_TP_INFO", DistributedInfo(0, 3))
    monkeypatch.setattr(
        embedding,
        "get_global_ctx",
        lambda: SimpleNamespace(batch=SimpleNamespace(size=2, is_prefill=False)),
    )
    head = ParallelLMHead(31, 8)
    head.weight = shards[0]
    head._comm = SimpleNamespace(all_gather=lambda _: gathered)
    torch.testing.assert_close(head.forward(x), F.linear(x, weight))


def _cpu_case(rank, config):
    import minisgl.models.qwen3_5_attn as attn_impl
    import minisgl.models.qwen3_5_delta as delta_impl

    torch.manual_seed(217)
    info._TP_INFO = DistributedInfo(0, 1)
    full = _modules(config)
    for module in full.values():
        for weight in module.state_dict().values():
            weight.normal_(0, 0.1)
    checkpoint = _checkpoint(full)
    x, decode_x = torch.randn(3, config.hidden_size), torch.randn(1, config.hidden_size)
    delta_impl._fla_chunk = delta_impl._fla_recurrent = None

    def prepare(modules):
        attention, delta, mlp = modules.values()
        for norm in (attention.q_norm, attention.k_norm):
            norm.rmsnorm = lambda x, w, eps: (
                x * (x.square().mean(-1, keepdim=True) + eps).rsqrt() * (1 + w)
            )
        mlp.act_fn = lambda x: F.silu(x.chunk(2, -1)[0]) * x.chunk(2, -1)[1]

        class CPUAttention:
            def forward(self, q, k, v, *args):
                q = q.view(1, -1, attention.num_qo_heads, config.head_dim).transpose(1, 2)
                k = k.view(1, -1, attention.num_kv_heads, config.head_dim).transpose(1, 2)
                v = v.view_as(k.transpose(1, 2)).transpose(1, 2)
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
                return out.transpose(1, 2).reshape(-1, attention.qo_dim)

        request = SimpleNamespace(cached_len=0, extend_len=3, table_idx=0)
        batch = SimpleNamespace(
            is_mixed=False,
            is_prefill=True,
            reqs=[request],
            attn_metadata=SimpleNamespace(shared_cache_op=CPUAttention()),
        )
        context = SimpleNamespace(
            batch=batch,
            gdn_ar=None,
            gdn_state=GDNStatePool(config, 1, torch.device("cpu"), torch.float32),
        )
        attn_impl.get_global_ctx = delta_impl.get_global_ctx = lambda: context
        return context

    context = prepare(full)
    expected_attention = full["model.layers.0.self_attn"].forward(x)
    expected_mlp = full["model.layers.0.mlp"].forward(x)
    expected_prefill = full["model.layers.0.linear_attn"].forward(x)
    context.batch.is_prefill = False
    expected_decode = full["model.layers.0.linear_attn"].forward(decode_x)
    expected_state = context.gdn_state.recurrent_state.clone()

    info._TP_INFO = DistributedInfo(rank, 2)
    local = _modules(config)
    _load(local, checkpoint, config, rank, 2)
    context = prepare(local)
    torch.testing.assert_close(local["model.layers.0.self_attn"].forward(x), expected_attention)
    torch.testing.assert_close(local["model.layers.0.mlp"].forward(x), expected_mlp)
    torch.testing.assert_close(local["model.layers.0.linear_attn"].forward(x), expected_prefill)
    context.batch.is_prefill = False
    torch.testing.assert_close(
        local["model.layers.0.linear_attn"].forward(decode_x), expected_decode
    )
    torch.testing.assert_close(
        context.gdn_state.recurrent_state, expected_state.chunk(2, dim=2)[rank]
    )


def _cpu_worker(rank, rendezvous):
    """Exercise equal shards and replicated KV heads over real collectives."""
    import torch.distributed as dist

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=60)
    )
    try:
        _cpu_case(rank, _config())
        _cpu_case(
            rank,
            replace(
                _config(),
                num_qo_heads=8,
                num_kv_heads=1,
                linear_num_key_heads=2,
                linear_num_value_heads=6,
            ),
        )
    finally:
        dist.destroy_process_group()


def test_two_rank_numerical_parity(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(_cpu_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2, join=True)


def test_server_cli_preserves_tp_request_and_derives_distributed_address():
    from minisgl.server.args import parse_args

    config, run_shell = parse_args(
        ["--model", "unused", "--dtype", "bfloat16", "--tp-size", "3", "--port", "2345"]
    )
    assert config.tp_info == DistributedInfo(0, 3)
    assert config.distributed_addr == "tcp://127.0.0.1:2346"
    assert replace(config, server_port=3456).distributed_addr == "tcp://127.0.0.1:3457"
    assert not run_shell


@pytest.mark.parametrize(
    "requested,available,expected",
    [(1, 1, 1), (2, 2, 2), (2, 6, 2), (4, 6, 4), (8, 8, 8)],
)
def test_server_starts_supported_requested_ranks(monkeypatch, requested, available, expected):
    from unittest.mock import Mock

    from minisgl.engine import EngineConfig
    from minisgl.server import api_server, args, launch

    config = args.ServerArgs(
        model_path="unused", tp_info=DistributedInfo(0, requested), dtype=torch.bfloat16
    )
    monkeypatch.setattr(EngineConfig, "model_config", property(lambda _: _config(large=True)))
    monkeypatch.setattr(args, "parse_args", lambda *_: (config, False))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: available)
    monkeypatch.setattr(launch.mp, "set_start_method", Mock())
    monkeypatch.setattr(launch.mp, "Queue", Mock())
    processes = Mock()
    monkeypatch.setattr(launch.mp, "Process", processes)
    logger = Mock()
    monkeypatch.setattr(launch, "init_logger", lambda *_: logger)

    def run_api(resolved, start_subprocess, **kwargs):
        assert resolved.tp_info == DistributedInfo(0, expected)
        assert processes.call_count == 0
        start_subprocess()

    monkeypatch.setattr(api_server, "run_api_server", run_api)
    launch.launch_server()
    schedulers = [
        call.kwargs["args"][0]
        for call in processes.call_args_list
        if call.kwargs["target"] is launch._run_scheduler
    ]
    assert [a.tp_info for a in schedulers] == [
        DistributedInfo(rank, expected) for rank in range(expected)
    ]
    assert all(a.distributed_addr == config.distributed_addr for a in schedulers)
    assert processes.call_count == expected + 1  # existing detokenizer process
    assert processes.return_value.start.call_count == expected + 1
    assert config.tp_info.size == requested
    logger.warning.assert_not_called()


@pytest.mark.parametrize(
    "requested,available,supported",
    [(3, 3, 2), (6, 6, 4), (16, 16, 8), (3, 1, 1), (8, 2, 2)],
)
def test_server_rejects_smaller_tp_before_startup(monkeypatch, requested, available, supported):
    from unittest.mock import Mock

    from minisgl.engine import EngineConfig
    from minisgl.server import api_server, args, launch

    config = args.ServerArgs(
        model_path="unused", tp_info=DistributedInfo(0, requested), dtype=torch.bfloat16
    )
    monkeypatch.setattr(EngineConfig, "model_config", property(lambda _: _config(large=True)))
    monkeypatch.setattr(args, "parse_args", lambda *_: (config, False))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: available)
    run_api, processes = Mock(), Mock()
    monkeypatch.setattr(api_server, "run_api_server", run_api)
    monkeypatch.setattr(launch.mp, "Process", processes)
    with pytest.raises(RuntimeError) as error:
        launch.launch_server()
    assert f"Requested TP={requested}, visible GPUs={available}" in str(error.value)
    assert f"Restart with --tp-size {supported}" in str(error.value)
    assert config.tp_info == DistributedInfo(0, requested)
    run_api.assert_not_called()
    processes.assert_not_called()


@pytest.mark.parametrize("available,invalid_model", [(0, False), (2, True)])
def test_server_rejects_invalid_layout_before_startup(monkeypatch, available, invalid_model):
    from unittest.mock import Mock

    from minisgl.engine import EngineConfig
    from minisgl.server import api_server, args, launch

    model = _config(large=True)
    if invalid_model:
        model = replace(model, intermediate_size=129, weight_block_size=128)
    config = args.ServerArgs(
        model_path="unused", tp_info=DistributedInfo(0, 2), dtype=torch.bfloat16
    )
    monkeypatch.setattr(EngineConfig, "model_config", property(lambda _: model))
    monkeypatch.setattr(args, "parse_args", lambda *_: (config, False))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: available)
    run_api, processes = Mock(), Mock()
    monkeypatch.setattr(api_server, "run_api_server", run_api)
    monkeypatch.setattr(launch.mp, "Process", processes)
    with pytest.raises(ValueError):
        launch.launch_server()
    run_api.assert_not_called()
    processes.assert_not_called()


def test_page_capacity_uses_local_memory_and_global_min(monkeypatch):
    import minisgl.engine.engine as engine_impl

    config = _config()
    engine = SimpleNamespace(
        device=torch.device("cpu"),
        dtype=torch.float32,
        tp_cpu_group=object(),
        _sync_get_memory=lambda: (100, 900),
    )
    # Local available memory = 0.9*10000 - (10000-8000) = 7000 bytes.
    monkeypatch.setattr(engine_impl, "get_free_memory", lambda _: 8000)

    def reduce_capacity(value, **kwargs):
        assert int(value) == 7000 // (2 * config.head_dim * 2 * 4)
        value.fill_(17)  # another rank can hold only 17 pages

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce_capacity)
    options = SimpleNamespace(
        model_config=config,
        tp_info=DistributedInfo(0, 2),
        page_size=1,
        num_page_override=None,
        memory_ratio=0.9,
    )
    assert engine_impl.Engine._determine_num_pages(engine, 10000, options) == 17
