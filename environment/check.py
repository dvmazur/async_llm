"""Explicit Qwen/Craftium environment check. Not part of normal run startup."""
def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=('minisgl', 'sglang'), default='minisgl')
    args = parser.parse_args(argv)
    import json
    import shutil
    if shutil.which('ninja') is None:
        raise RuntimeError('ninja is not on PATH; include the selected venv/bin before loading a model')
    import torch
    from flashinfer import BatchPrefillWithPagedKVCacheWrapper
    import craftium
    import mt_server
    if not callable(getattr(craftium, 'CraftiumEnv', None)):
        raise RuntimeError('Craftium import is shadowed by a checkout directory; rerun Setup to install its compatible editable path')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    if args.backend == 'sglang':
        import sglang
        import sgl_kernel
        print(json.dumps(dict(backend='sglang', sglang=sglang.__version__,
            torch=torch.__version__, gpu=torch.cuda.get_device_name(), craftium=craftium.__file__)))
        return
    import minisgl.models.qwen3_5_delta as delta
    capability = torch.cuda.get_device_capability()
    if capability != (12, 1):
        import sgl_kernel  # missing shared libraries must fail before a costly run
    else:
        import warnings
        warnings.warn('SM121: intentional engine MoE compatibility path; not native RTX control')
    if delta._fla_chunk is None or delta._fla_recurrent is None:
        raise RuntimeError('FLA failed to import; refusing a silent slow control')
    if not hasattr(BatchPrefillWithPagedKVCacheWrapper, 'workspace_size'):
        raise RuntimeError('FlashInfer lacks required graph workspace API')
    print(json.dumps(dict(gpu=torch.cuda.get_device_name(), capability=capability,
        torch=torch.__version__, cuda=torch.version.cuda, mt_server=mt_server.__file__,
        craftium=craftium.__file__, engine=delta.__file__)), flush=True)


if __name__ == '__main__':
    main()
