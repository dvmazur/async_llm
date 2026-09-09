"""Frozen old compose method, with local snapshots of its two math helpers.

Source: fp8/tests/core/test_gdn_compose.py (pre-planned runtime).
The method body below is literal. Do not import changed production helpers.
"""
import torch


def init_gdn_affine(*, batch_size, num_heads, d_k, d_v, dtype, device):
    eye = torch.eye(d_k, dtype=dtype, device=device)
    A_hat = eye.view(1, 1, d_k, d_k).expand(batch_size, num_heads, d_k, d_k).clone()
    B_hat = torch.zeros(batch_size, num_heads, d_v, d_k, dtype=dtype, device=device)
    return A_hat, B_hat


def compose_gdn_affines(*, A_first, B_first, A_second, B_second):
    A = torch.matmul(A_first, A_second)
    B = torch.matmul(B_first, A_second)
    B.add_(B_second)
    return A, B


def _reference_compose_initial_recurrent_state(
    self, lin_idx: int, dtype: torch.dtype
) -> torch.Tensor | None:
    """Compose each worker's chain into an initial recurrent state.

    Returns ``[num_workers, H, d_k, d_v]`` in HF convention (or ``None`` if no
    block in any chain has an affine for this layer).
    """
    if not self.has_previous_affine(lin_idx):
        return None

    # Worker chains typically share leading blocks (e.g. [prompt, thinker] is a
    # prefix of [prompt, thinker, writer]).  Memoize each composed prefix by its
    # block-id tuple so a shared prefix is composed once per call, not per worker.
    # Bit-identical to composing each chain independently (compose is deterministic).
    prefix_memo: dict = {}

    def compose_chain(chain):
        acc = None  # (A, B) once we hit the first block with an affine
        key: tuple = ()
        for block in chain:
            key = key + (id(block),)
            pair = block.linear_affine.get(lin_idx)
            if pair is None:
                continue  # block has no affine for this layer -> acc unchanged
            if key in prefix_memo:
                acc = prefix_memo[key]
                continue
            A_b = pair[0].to(dtype=torch.float32, device=self.device)
            B_b = pair[1].to(dtype=torch.float32, device=self.device)
            if acc is None:
                acc = (A_b, B_b)  # first real block: no identity compose needed
            else:
                acc = compose_gdn_affines(
                    A_first=acc[0], B_first=acc[1], A_second=A_b, B_second=B_b
                )
            prefix_memo[key] = acc
        if acc is None:
            acc = init_gdn_affine(
                batch_size=1,
                num_heads=self.num_heads,
                d_k=self.head_k_dim,
                d_v=self.head_v_dim,
                dtype=torch.float32,
                device=self.device,
            )
        return acc

    per_worker = [compose_chain(chain)[1] for chain in self.cache_structure]
    S_block = torch.cat(per_worker, dim=0)  # [W, H, d_v, d_k]
    S_hf = S_block.transpose(-1, -2).contiguous()  # [W, H, d_k, d_v]
    return S_hf.to(dtype=dtype)
