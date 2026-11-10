"""The two serialization schemas accepted by the existing block-FP8 kernels.

This is a checkpoint adapter, not a compressed-tensors execution backend.
Both schemas store E4M3 values and multiplicative FP32 scales per 128x128 block.
"""


def validate_fp8_config(config: dict) -> None:
    error = "Expected serialized dynamic E4M3 weights with 128x128 block scales"
    if not isinstance(config, dict):
        raise ValueError(error)
    if config.get("quant_method") == "fp8":
        if (config.get("fmt", "e4m3") == "e4m3"
                and config.get("activation_scheme") == "dynamic"
                and config.get("weight_block_size") == [128, 128]):
            return
    elif config.get("quant_method") == "compressed-tensors":
        groups = config.get("config_groups") or {}
        if (config.get("format") != "float-quantized"
                or config.get("quantization_status") != "compressed"
                or config.get("kv_cache_scheme") or config.get("transform_config")
                or config.get("sparsity_config") or len(groups) != 1):
            raise ValueError(error)
        group = next(iter(groups.values()))
        w, a = group.get("weights") or {}, group.get("input_activations") or {}
        common = all(x.get("num_bits") == 8 and x.get("type") == "float"
                     and x.get("symmetric") is True for x in (w, a))
        if (common and group.get("targets") == ["Linear"]
                and group.get("output_activations") is None
                and w.get("strategy") == "block" and w.get("block_structure") == [128, 128]
                and w.get("dynamic") is False
                and a.get("strategy") == "group" and a.get("group_size") == 128
                and a.get("dynamic") is True):
            return
    raise ValueError(error)
