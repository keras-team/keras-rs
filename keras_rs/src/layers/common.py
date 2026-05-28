import keras
from keras import ops
from typing import List, Optional, Tuple

def fx_unwrap_optional_tensor(optional: Optional[keras.KerasTensor]) -> keras.KerasTensor:
    """Helper to unwrap optional tensors, returning a zero-tensor for uninitialized cache."""
    if optional is None:
        # Returning a zero-tensor is necessary for graph tracing when the cache is uninitialized.
        return ops.zeros((0,), dtype='float32') 
    return optional

def get_valid_attn_mask_keras(
    causal: bool,
    N: int,
    seq_lengths: keras.KerasTensor,
    num_targets: Optional[keras.KerasTensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    min_full_attn_seq_len: int = 0,
) -> keras.KerasTensor:
    """
    Keras implementation of the valid attention mask generation, combining
    causality, sequence lengths, and target awareness.
    """
    ids = ops.reshape(ops.arange(0, N, dtype="int32"), (1, N))
    max_ids = ops.reshape(seq_lengths, (-1, 1, 1))
    B = ops.shape(seq_lengths)[0]

    if contextual_seq_len > 0:
        ids = ids - contextual_seq_len + 1
        ids = ops.maximum(ids, 0)
        max_ids = max_ids - contextual_seq_len + 1

    if num_targets is not None:
        max_ids = max_ids - ops.reshape(num_targets, (-1, 1, 1))
        ids = ops.minimum(ids, max_ids)
        row_ids = ops.broadcast_to(ops.reshape(ids, (-1, N, 1)), (B, N, N))
        col_ids = ops.broadcast_to(ops.reshape(ids, (-1, 1, N)), (B, N, N))
    else:
        row_ids = ops.broadcast_to(ops.reshape(ids, (N, 1)), (N, N))
        col_ids = ops.transpose(row_ids)
        row_ids = ops.reshape(row_ids, (1, N, N))
        col_ids = ops.reshape(col_ids, (1, N, N))
        max_ids = None

    row_col_dist = row_ids - col_ids
    valid_attn_mask = ops.reshape(ops.eye(N, dtype="bool"), (1, N, N))

    if not causal:
        row_col_dist = ops.where(row_col_dist > 0, row_col_dist, -row_col_dist)

    valid_attn_mask = ops.logical_or(valid_attn_mask, row_col_dist > 0)

    if max_attn_len > 0:
        valid_attn_mask = ops.logical_and(valid_attn_mask, row_col_dist <= max_attn_len)

    if contextual_seq_len > 0 and max_ids is not None:
        valid_attn_mask = ops.logical_or(
            valid_attn_mask, ops.logical_and(row_ids == 0, col_ids < max_ids)
        )

    return valid_attn_mask
