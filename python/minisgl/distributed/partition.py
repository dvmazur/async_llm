"""Validate the existing equal TP shards and choose a smaller supported size."""

from minisgl.utils import div_even


def validate_model_partition(config, size: int) -> None:
    """Reject partial heads, unequal shards and cuts through serialized FP8 blocks."""
    if size < 1:
        raise ValueError("tensor_parallel_size must be positive")
    try:
        div_even(config.num_qo_heads, config.num_kv_heads)
        widths = [
            div_even(config.num_qo_heads, size) * config.head_dim,
            div_even(config.num_kv_heads, size, allow_replicate=True) * config.head_dim,
        ]
        if config.is_hybrid:
            div_even(config.linear_num_value_heads, config.linear_num_key_heads)
            widths += [
                div_even(config.linear_num_key_heads, size) * config.linear_key_head_dim,
                div_even(config.linear_num_value_heads, size) * config.linear_value_head_dim,
            ]
        intermediate_sizes = (
            [config.moe_intermediate_size, config.shared_expert_intermediate_size]
            if config.is_moe
            else [config.intermediate_size]
        )
        widths += [div_even(width, size) for width in intermediate_sizes if width]
        if any(width % config.weight_block_size for width in widths):
            raise ValueError("shards cut through serialized FP8 blocks")
    except (AssertionError, ValueError) as exc:
        raise ValueError(f"TP={size} does not support equal shards for this model: {exc}") from exc


def select_tp_size(config, requested: int, available: int) -> int:
    """Largest supported TP no greater than the request and visible GPU count."""
    if requested < 1:
        raise ValueError("tensor_parallel_size must be positive")
    if available < 1:
        raise ValueError("No visible CUDA devices")
    for size in range(min(requested, available, config.num_qo_heads), 0, -1):
        try:
            validate_model_partition(config, size)
        except ValueError:
            continue
        return size
    raise ValueError("This model has no supported equal TP partition")
