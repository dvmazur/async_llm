"""Load existing BF16/block-FP8 Qwen weights without old Engine's extra pools."""
from pathlib import Path
import torch


@torch.inference_mode()
def load_model(model_path,*,device="cuda:0",dtype=torch.bfloat16):
    """Use the original config/weight merger and precision contracts unchanged.

    Downloads and environment installation are intentionally not side effects
    of this helper: provide an existing local checkpoint/snapshot directory.
    It allocates weights only. PlannedSession subsequently owns all runtime
    pools/profile buffers; no serving KV/GDN pool or old graph runner exists.
    """
    from minisgl.distributed import try_get_tp_info,set_tp_info
    from minisgl.models import create_model,load_weight
    from minisgl.models.config import ModelConfig
    from minisgl.utils import cached_load_hf_config,torch_dtype
    if not Path(model_path).is_dir():raise ValueError("provide a downloaded local checkpoint directory")
    device=torch.device(device)
    if device.type!="cuda" or dtype not in (torch.bfloat16,torch.float16):
        raise ValueError("planned loader requires CUDA BF16/FP16 activation dtype")
    tp=try_get_tp_info()
    if tp is None:set_tp_info(0,1)
    elif tp.size!=1:raise ValueError("planned loader supports TP1; refusing to modify an existing TP context")
    config=ModelConfig.from_hf(cached_load_hf_config(str(model_path)))
    if not config.is_hybrid:raise ValueError("planned runtime requires the Qwen hybrid architecture")
    with torch.cuda.device(device),torch.device("meta"),torch_dtype(dtype):model=create_model(config)
    state={name:(value if value.dtype==torch.float8_e4m3fn else value.float() if name.endswith("_scale_inv")
                 else value.to(dtype)) for name,value in load_weight(str(model_path),device)}
    model.load_state_dict(state)
    if any(t.is_meta for t in model.state_dict().values()):raise RuntimeError("checkpoint left unmaterialized weights")
    return model
