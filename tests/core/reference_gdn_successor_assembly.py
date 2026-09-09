"""Literal pre-optimization begin_decode_state, retained only as a test oracle."""
import torch
from minisgl.shared_cache.gdn_successor_cache import StateBatch, StateRow, assemble_state_rows

def begin_decode_state(self, lin_idx):
    """Return a v-first initial state and a pre-write validation ticket.

    Only this decode entry point consults successors. Prefill always uses
    the original compose API; its writes naturally invalidate saved keys.
    """
    cache = self.successor_state_cache
    if cache is None or self.prefill_segments is not None:
        return self.compose_initial_recurrent_state(
            lin_idx, torch.float32, state_v_first=True
        ), None

    write_ids = [id(block) for block in self.write_to]
    write_set = set(write_ids)
    tickets, rows = [], []
    for chain, target in zip(self.cache_structure, self.write_to):
        eligible = (
            chain and chain[-1] is target
            and write_ids.count(id(target)) == 1
            and all(id(block) not in write_set for block in chain[:-1])
            and all(hasattr(block, "linear_affine_revision") for block in chain)
        )
        signature = self._decode_signature(chain, lin_idx) if eligible else None
        tickets.append(None if signature is None else (id(target), signature))
        rows.append(None if signature is None else cache.get(target, lin_idx, signature))

    missing = [w for w, row in enumerate(rows) if row is None]
    if len(missing) == self.num_workers:
        return self.compose_initial_recurrent_state(
            lin_idx, torch.float32, state_v_first=True
        ), tickets
    if missing:
        view = self.context_view(
            [self.cache_structure[w] for w in missing], [self.write_to[w] for w in missing]
        )
        state = view.compose_initial_recurrent_state(lin_idx, torch.float32, state_v_first=True)
        if state is None:
            state = torch.zeros(
                len(missing), self.num_heads, self.head_v_dim, self.head_k_dim,
                dtype=torch.float32, device=self.device,
            )
        batch = StateBatch(state)
        for index, worker in enumerate(missing):
            rows[worker] = StateRow(batch, index, (), None)
    return assemble_state_rows(rows), tickets
