import keras
from keras import ops
from typing import Tuple, Optional
from keras import layers

# --- Assumed Imports ---
# Assumes keras_jagged_to_padded_dense, keras_dense_to_jagged, and get_valid_attn_mask_keras are available from other modules.

def keras_pad_qkv(
    q: keras.KerasTensor, k: keras.KerasTensor, v: keras.KerasTensor, seq_offsets: keras.KerasTensor, N: int,
) -> Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]:
    """
    Helper to pad Q, K, V from jagged to dense format for MHA.
    Assumes keras_jagged_to_padded_dense is available globally.
    """
    L, H, D = ops.shape(q); V_dim = ops.shape(v)[2]
    values_q = ops.reshape(q, [L, H * D]); values_k = ops.reshape(k, [L, H * D]); values_v = ops.reshape(v, [L, H * V_dim])
    
    # Pad Q, K, V
    padded_q = keras_jagged_to_padded_dense(values=values_q, offsets=[seq_offsets], max_lengths=[N], padding_value=0.0)
    padded_k = keras_jagged_to_padded_dense(values=values_k, offsets=[seq_offsets], max_lengths=[N], padding_value=0.0)
    padded_v = keras_jagged_to_padded_dense(values=values_v, offsets=[seq_offsets], max_lengths=[N], padding_value=0.0)

    B = ops.shape(padded_q)[0]
    padded_q = ops.reshape(padded_q, [B, N, H, D]); padded_k = ops.reshape(padded_k, [B, N, H, D]); padded_v = ops.reshape(padded_v, [B, N, H, V_dim])
    padded_q = ops.transpose(padded_q, [0, 2, 1, 3]); padded_k = ops.transpose(padded_k, [0, 2, 1, 3])
    padded_v = ops.transpose(padded_v, [0, 2, 1, 3])
    return padded_q, padded_k, padded_v


def keras_hstu_mha(
    max_seq_len: int, alpha: float, q: keras.KerasTensor, k: keras.KerasTensor, v: keras.KerasTensor, seq_offsets: keras.KerasTensor, causal: bool = True, dropout_pr: float = 0.0, training: bool = True, attn_scale: Optional[keras.KerasTensor] = None, **kwargs
) -> keras.KerasTensor:
    """Core Keras implementation of the full Multi-Head Attention kernel (Non-Cached)."""
    L, H, _ = ops.shape(q); V_dim = ops.shape(v)[2]
    q, k, v = keras_pad_qkv(q, k, v, seq_offsets, max_seq_len)
    qk_attn = ops.einsum("bhxa,bhya->bhxy", q, k) * alpha

    # Activation and Scaling
    if attn_scale is not None:
        if ops.ndim(attn_scale) > 0:
            attn_scale_padded = keras_jagged_to_padded_dense(values=ops.expand_dims(attn_scale, axis=-1), offsets=[seq_offsets], max_lengths=[max_seq_len], padding_value=0.0)
            attn_scale_padded = ops.expand_dims(ops.cast(attn_scale_padded, qk_attn.dtype), axis=1)
        qk_attn = ops.silu(qk_attn) * attn_scale_padded
    else:
        qk_attn = ops.silu(qk_attn) / max_seq_len

    # Masking
    seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
    valid_attn_mask = get_valid_attn_mask_keras(causal=causal, N=max_seq_len, seq_lengths=seq_lengths, **kwargs)
    qk_attn = qk_attn * ops.expand_dims(ops.cast(valid_attn_mask, qk_attn.dtype), axis=1)

    # Dropout
    if dropout_pr > 0.0 and training:
        qk_attn = keras.layers.Dropout(dropout_pr)(qk_attn, training=training)

    # Output (Weighted Sum)
    attn_dense = ops.einsum("bhxd,bhdv->bhxv", qk_attn, v)
    flat_attn_dense = ops.reshape(ops.transpose(attn_dense, [0, 2, 1, 3]), [-1, max_seq_len, H * V_dim])

    # Convert back to jagged
    jagged_output = keras_dense_to_jagged(flat_attn_dense, [seq_offsets])
    L_out = ops.shape(jagged_output)[0]
    return ops.reshape(jagged_output, [L_out, H, V_dim])


def keras_cached_hstu_mha(
    max_seq_len: int, alpha: float, delta_q: keras.KerasTensor, k: keras.KerasTensor, v: keras.KerasTensor, seq_offsets: keras.KerasTensor, num_targets: Optional[keras.KerasTensor] = None, max_attn_len: int = 0, contextual_seq_len: int = 0, enable_tma: bool = False,
) -> keras.KerasTensor:
    """Core Keras implementation of the cached attention kernel (Delta Q attends to Full K/V)."""
    L_delta, H, D = ops.shape(delta_q); B = ops.shape(seq_offsets)[0] - 1; DeltaSize = L_delta // B; V_dim = ops.shape(v)[2]

    # 1. Reshape Delta Q
    delta_q = ops.transpose(ops.reshape(delta_q, (B, DeltaSize, H, D)), perm=[0, 2, 1, 3])
    
    # 2. Reshape Full K and V (Inputs k, v are already flattened/jagged-like)
    N_full = max_seq_len
    k_full = ops.transpose(ops.reshape(k, (B, N_full, H, D)), [0, 2, 1, 3])
    v_full = ops.transpose(ops.reshape(v, (B, N_full, H, V_dim)), [0, 2, 1, 3])

    # 3. Attention Score and Activation
    qk_attn = ops.einsum("bhxa,bhya->bhxy", delta_q, k_full) * alpha
    qk_attn = ops.silu(qk_attn) / max_seq_len

    # 4. Masking (Slice the mask to select only the rows corresponding to the new queries)
    seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
    full_valid_attn_mask = get_valid_attn_mask_keras(causal=True, N=max_seq_len, seq_lengths=seq_lengths, num_targets=num_targets, max_attn_len=max_attn_len, contextual_seq_len=contextual_seq_len)
    valid_attn_mask_sliced = full_valid_attn_mask[:, -DeltaSize:, :] 

    qk_attn = qk_attn * ops.expand_dims(ops.cast(valid_attn_mask_sliced, qk_attn.dtype), axis=1)

    # 5. Output (Weighted Sum)
    attn_output = ops.einsum("bhxd,bhdv->bhxv", qk_attn, v_full)

    # 6. Reshape and return [L_delta, H, V_dim]
    attn_output = ops.transpose(attn_output, perm=[0, 2, 1, 3]) 
    return ops.reshape(attn_output, (-1, H, V_dim))


def delta_hstu_mha(
    max_seq_len: int, alpha: float, delta_q: keras.KerasTensor, k: keras.KerasTensor, v: keras.KerasTensor, seq_offsets: keras.KerasTensor, num_targets: Optional[keras.KerasTensor] = None, max_attn_len: int = 0, contextual_seq_len: int = 0, kernel=None, enable_tma: bool = False,
) -> keras.KerasTensor:
    """Top-level wrapper for cached inference MHA (delegates to core cached kernel)."""
    
    L_delta, H, D = ops.shape(delta_q)
    # Assertions are maintained by the layer/framework where possible.
    
    return keras_cached_hstu_mha(
        max_seq_len=max_seq_len, alpha=alpha, delta_q=delta_q, k=k, v=v, seq_offsets=seq_offsets,
        num_targets=num_targets, max_attn_len=max_attn_len, contextual_seq_len=contextual_seq_len,
    )
