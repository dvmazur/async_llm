"""Architecture and backward requirements of the supported FA4 replay paths."""

import os


def fa4_unsupported_reason(capability: tuple[int, int], head_size: int) -> str | None:
    if capability[0] != 9 and capability != (10, 3):
        return "FA4 replay requires Hopper (SM90) or B300 (SM103)"
    if not 8 <= head_size <= 256 or head_size % 8:
        return "FA4 replay requires a head dimension divisible by 8 between 8 and 256"
    if capability == (10, 3):
        if 128 < head_size < 256:
            return "B300 FA4 replay requires a head dimension between 8 and 128, or 256"
        # FA4 4.0.0b37's dedicated SM100 HD256 backward rejects dLSE, which
        # replay's partial-output merge needs. FA4 reads this flag at backward;
        # leave it set for the entire training process, including recomputation.
        if head_size == 256 and os.environ.get("FLASH_ATTENTION_HD256_GENERIC_BWD") != "1":
            return (
                "B300 FA4 replay with head dimension 256 requires "
                "FLASH_ATTENTION_HD256_GENERIC_BWD=1 for the entire training process "
                "to preserve LSE gradients"
            )
    return None
