"""Low-overhead topology and CUDA timing trace for SharedCacheGDN.

Only one representative GDN layer is traced.  All Qwen linear-attention layers
share the same cache topology and state shape, so recording every layer would add
30x as many CUDA events without adding policy information.
"""

from __future__ import annotations

import json
import statistics
import time
import weakref
from collections import defaultdict
from pathlib import Path

import torch


class ComposeCacheTrace:
    METHOD_NAMES = (
        "prior_conv_states",
        "compose_initial_recurrent_state",
        "capture_token_affines",
        "set_conv_states",
    )

    def __init__(self, output_path: Path, representative_layer: int = 0):
        self.output_path = Path(output_path)
        self.representative_layer = int(representative_layer)
        self.events: list[dict[str, object]] = []
        self._timings: list[tuple[str, str, torch.cuda.Event, torch.cuda.Event, float]] = []
        self._pair_versions: dict[
            tuple[int, int],
            tuple[weakref.ReferenceType, weakref.ReferenceType, int],
        ] = {}
        self._originals: dict[str, object] = {}
        self._installed = False
        self._finalized = False

    def _phase(self, gdn) -> str:
        return "decode" if gdn.prefill_segments is None else "prefill"

    def _version(self, block, lin_idx: int) -> int:
        pair = block.linear_affine.get(lin_idx)
        if pair is None:
            raise ValueError("version requested for an affine-empty block")
        key = (int(block.block_id), int(lin_idx))
        prior = self._pair_versions.get(key)
        if prior is not None and prior[0]() is pair[0] and prior[1]() is pair[1]:
            return prior[2]
        version = 1 if prior is None else prior[2] + 1
        self._pair_versions[key] = (weakref.ref(pair[0]), weakref.ref(pair[1]), version)
        return version

    def _record_topology(self, gdn, lin_idx: int) -> None:
        chains = []
        effective_lengths = []
        for chain in gdn.cache_structure:
            effective = []
            for block in chain:
                if block.linear_affine.get(lin_idx) is None:
                    continue
                effective.append([int(block.block_id), self._version(block, lin_idx)])
            chains.append(effective)
            effective_lengths.append(len(effective))
        self.events.append(
            {
                "index": len(self.events),
                "phase": self._phase(gdn),
                "workers": len(chains),
                "effective_chain_lengths": effective_lengths,
                "chains": chains,
                "write_block_ids": [int(block.block_id) for block in gdn.write_to],
            }
        )

    def install(self, gdn) -> None:
        if self._installed:
            raise RuntimeError("trace is already installed")
        self._installed = True
        for name in self.METHOD_NAMES:
            original = getattr(gdn, name)
            self._originals[name] = original

            def wrapped(lin_idx, *args, __name=name, __original=original, **kwargs):
                traced = int(lin_idx) == self.representative_layer
                if traced and __name == "compose_initial_recurrent_state":
                    self._record_topology(gdn, int(lin_idx))
                if not traced:
                    return __original(lin_idx, *args, **kwargs)

                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                cpu_started = time.perf_counter()
                start.record()
                result = __original(lin_idx, *args, **kwargs)
                end.record()
                cpu_elapsed = time.perf_counter() - cpu_started
                self._timings.append((__name, self._phase(gdn), start, end, cpu_elapsed))
                return result

            setattr(gdn, name, wrapped)

    @staticmethod
    def _summary(values: list[float]) -> dict[str, float | int | None]:
        if not values:
            return {
                "calls": 0,
                "sum_ms": 0.0,
                "mean_ms": None,
                "median_ms": None,
                "p90_ms": None,
                "max_ms": None,
            }
        ordered = sorted(values)
        p90_index = min(len(ordered) - 1, int(0.9 * len(ordered)))
        return {
            "calls": len(values),
            "sum_ms": sum(values),
            "mean_ms": statistics.fmean(values),
            "median_ms": statistics.median(values),
            "p90_ms": ordered[p90_index],
            "max_ms": ordered[-1],
        }

    def finalize(self) -> Path:
        if self._finalized:
            return self.output_path
        self._finalized = True
        torch.cuda.synchronize()
        gpu: dict[tuple[str, str], list[float]] = defaultdict(list)
        cpu: dict[tuple[str, str], list[float]] = defaultdict(list)
        for name, phase, start, end, cpu_seconds in self._timings:
            gpu[(name, phase)].append(float(start.elapsed_time(end)))
            cpu[(name, phase)].append(cpu_seconds * 1000.0)
        timings = {}
        for key in sorted(set(gpu) | set(cpu)):
            name, phase = key
            timings[f"{phase}:{name}"] = {
                "gpu": self._summary(gpu[key]),
                "cpu_dispatch": self._summary(cpu[key]),
            }
        output = {
            "schema_version": 1,
            "representative_layer": self.representative_layer,
            "state_bytes_per_cached_prefix": 32 * 128 * 128 * 4,
            "num_gdn_layers": 30,
            "timings": timings,
            "events": self.events,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(output, indent=2) + "\n")
        print(f"COMPOSE_CACHE_TRACE {self.output_path}", flush=True)
        return self.output_path

    def uninstall(self, gdn) -> None:
        """Restore methods replaced by :meth:`install`."""
        if not self._installed:
            return
        for name, original in self._originals.items():
            setattr(gdn, name, original)
        self._installed = False


__all__ = ["ComposeCacheTrace"]
