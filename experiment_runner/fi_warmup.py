"""Per-process FlashInfer sampler initialization, not an episode or model forward."""


def warmup_sampling(device, vocab_size):
    # Import only after Engine has initialized CUDA in this process.
    import torch
    import flashinfer.sampling as sampling

    probabilities = torch.full((1, vocab_size), 1/vocab_size, dtype=torch.float32, device=device)
    generator = torch.Generator(device=device).manual_seed(0)
    sampling.top_k_top_p_sampling_from_probs(probabilities, 20, .9, generator=generator)
    torch.cuda.synchronize(device)
