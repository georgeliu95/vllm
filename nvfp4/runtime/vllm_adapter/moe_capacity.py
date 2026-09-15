"""Capacity policy for local MoE dispatch."""


def bounded_capacity(tokens: int, maximum: int) -> int:
    """Keep all input rows with at most one 128-row alignment block of padding."""
    return min(maximum, max(128, ((tokens + 127) // 128) * 128))
