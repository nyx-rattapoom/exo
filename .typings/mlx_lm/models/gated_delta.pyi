from typing import Optional

import mlx.core as mx

def compute_g(A_log: mx.array, a: mx.array, dt_bias: mx.array) -> mx.array: ...
def gated_delta_update(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    state: Optional[mx.array] = ...,
    mask: Optional[mx.array] = ...,
    use_kernel: bool = ...,
) -> tuple[mx.array, mx.array]: ...
def gated_delta_ops(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: Optional[mx.array] = ...,
    mask: Optional[mx.array] = ...,
) -> tuple[mx.array, mx.array]: ...

# The pinned fork (rltakashige/mlx-lm@leo/deepseek-v4) fuses the g/beta
# computation into the Metal kernel, so this takes the raw projections rather
# than upstream's precomputed (g, beta).
def gated_delta_kernel(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    state: mx.array,
    mask: Optional[mx.array] = ...,
) -> tuple[mx.array, mx.array]: ...
