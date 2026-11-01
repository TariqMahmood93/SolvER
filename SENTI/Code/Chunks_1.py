# Chunks_creation.py
# ------------------------------------------------------------
# This module: determines where each incremental “subset”
# should end, given a dataset length, an initial chunk size,
# and a constant step.
# ------------------------------------------------------------


def create_chunks(full_len: int, initial_size: int, step: int) -> list[int]:
    """Return the inclusive end-indices for successive incremental subsets."""
    ends = list(range(initial_size, full_len + 1, step))
    if not ends or ends[-1] != full_len:
        ends.append(full_len)
    return ends
