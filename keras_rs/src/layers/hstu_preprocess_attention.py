import keras
from keras import ops
from typing import Tuple, List, Optional


def keras_hstu_preprocess_and_attention(
    x: keras.KerasTensor, norm_weight: keras.KerasTensor, norm_bias: keras.KerasTensor, norm_eps: float, num_heads: int, attn_dim: int, hidden_dim: int,
    uvqk_weight: keras.KerasTensor, uvqk_bias: keras.KerasTensor, max_seq_len: int, seq_offsets: keras.KerasTensor, attn_alpha: float, causal: bool,
    num_targets: Optional[keras.KerasTensor], max_attn_len: int, contextual_seq_len: int, recompute_uvqk_in_backward: bool,
    recompute_normed_x_in_backward: bool, sort_by_length: bool, prefill: bool = False,
    kernel=None, **kwargs
) -> Tuple:
    """
    Keras 3 implementation of the H-STU preprocess and attention workflow.
    Orchestrates the conversion of input X into U, Q, K, V and subsequent MHA computation.
    """

    # --- Assertions (Skipped internal torch asserts, simplified to Keras asserts for context) ---
    assert max_seq_len > 0, "max_seq_len must be larger than 0"
    assert ops.ndim(x) == 2, "x must be 2-D"
    assert causal is True, "only causal attention is supported."

    # 1. Compute U, Q, K, V
    # Note: hstu_compute_uqvk handles the initial Norm, Linear Projection, and Split.
    u, q, k, v = hstu_compute_uqvk(
        x=x, norm_weight=norm_weight, norm_bias=norm_bias, norm_eps=norm_eps,
        num_heads=num_heads, attn_dim=attn_dim, hidden_dim=hidden_dim,
        uvqk_weight=uvqk_weight, uvqk_bias=uvqk_bias, kernel=kernel,
    )

    # 2. Compute Attention
    attn_output = keras_hstu_mha(
        max_seq_len=max_seq_len, alpha=attn_alpha, q=q, k=k, v=v,
        seq_offsets=seq_offsets, causal=causal, dropout_pr=0.0,
        training=False, num_targets=num_targets, max_attn_len=max_attn_len,
        contextual_seq_len=contextual_seq_len, sort_by_length=sort_by_length,
        kernel=kernel, **kwargs
    )

    # Reshape: [L, H, D] -> [L, H * D] (Flattening for the final hstu_compute_output block)
    attn_output = ops.reshape(attn_output, [-1, hidden_dim * num_heads])

    # Returns u (gating), attention output, k, and v (for caching)
    return u, attn_output, k, v
