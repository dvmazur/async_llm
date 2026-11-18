"""Collect training traces with corrections arriving during reasoning."""

from .rollout import collect_correction_example, validate_correction_example

__all__ = ["collect_correction_example", "validate_correction_example"]
