import keras
from keras import ops
from typing import List, Optional, Tuple

# --- Core Jagged/Dense Conversion Functions ---

def keras_jagged_to_padded_dense(values, offsets, max_lengths, padding_value=0.0):
    """
    Keras 3 implementation to convert jagged tensor (values) into a padded dense tensor [B, N, D_flat].
    Required by MHA kernel padding (keras_pad_qkv).
    """
    offsets = offsets[0] if isinstance(offsets, list) else offsets
    B = ops.shape(offsets)[0] - 1
    max_len = max_lengths[0]
    D_flat = ops.shape(values)[-1]
    if ops.shape(values)[0] == 0:
        return ops.full((B, max_len, D_flat), padding_value, dtype=values.dtype)

    def pad_one(i):
        start = offsets[i]; end = offsets[i+1]
        seq_len = end - start 
        seq = ops.slice(values, [start, 0], [seq_len, D_flat])
        if ops.equal(seq_len, 0):
             return ops.full((max_len, D_flat), padding_value, dtype=values.dtype)
        if seq_len < max_len:
            padding_shape = ops.stack([max_len - seq_len, D_flat])
            padding = ops.full(padding_shape, padding_value, dtype=values.dtype)
            return ops.concatenate([seq, padding], axis=0)
        else:
            return seq[:max_len]

    idxs = ops.arange(B, dtype='int32')
    return ops.map(pad_one, idxs)

def keras_dense_to_jagged(
    dense: keras.KerasTensor,
    x_offsets: List[keras.KerasTensor],
) -> keras.KerasTensor:
    """Keras 3 implementation to convert a padded dense tensor [B, N, D] back into a jagged tensor."""
    seq_offsets = x_offsets[0]
    N = ops.shape(dense)[1] 
    D_flat = ops.shape(dense)[2] 
    token_range = ops.arange(N)
    seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
    mask = ops.expand_dims(token_range, axis=0) < ops.expand_dims(seq_lengths, axis=1)
    
    flattened = ops.reshape(dense, [-1, D_flat])
    flattened_mask = ops.reshape(mask, [-1])

    return flattened[flattened_mask]

# --- Jagged Splitting and Concatenation Wrappers (Used by Caching Logic) ---

def split_2D_jagged(
    max_seq_len: int, values: keras.KerasTensor, total_len_left: Optional[int] = None, total_len_right: Optional[int] = None, max_len_left: Optional[int] = None, max_len_right: Optional[int] = None, offsets_left: Optional[keras.KerasTensor] = None, offsets_right: Optional[keras.KerasTensor] = None, kernel=None,
) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
    """Top-level wrapper for splitting a concatenated jagged tensor."""

    def keras_split_2D_jagged_jagged(max_seq_len, values, offsets_left, offsets_right):
        D_flat = ops.shape(values)[1]; offsets = offsets_left + offsets_right
        padded_values_bnd = keras_jagged_to_padded_dense(values=values, offsets=[offsets], max_lengths=[max_seq_len], padding_value=0.0)
        padded_values = ops.reshape(padded_values_bnd, [-1, D_flat])
        lengths_left = offsets_left[1:] - offsets_left[:-1]; lengths_right = offsets_right[1:] - offsets_right[:-1]
        mask = ops.reshape(ops.arange(max_seq_len, dtype='int32'), [1, -1])
        lengths_left_broadcast = ops.reshape(lengths_left, [-1, 1]); lengths_right_combined = ops.reshape(lengths_left + lengths_right, [-1, 1])
        mask_left = mask < lengths_left_broadcast
        mask_right = ops.logical_and(mask >= lengths_left_broadcast, mask < lengths_right_combined)
        return padded_values[ops.reshape(mask_left, [-1])], padded_values[ops.reshape(mask_right, [-1])]

    def keras_split_2D_jagged_resolver(max_seq_len, values, max_len_left, max_len_right, offsets_left, offsets_right):
        L_total = ops.shape(values)[0]
        offsets_left_non_optional = offsets_left
        if offsets_left is None: offsets_left_non_optional = max_len_left * ops.arange(L_total // max_len_left + 1, dtype='int32')
        offsets_right_non_optional = offsets_right
        if offsets_right is None: offsets_right_non_optional = max_len_right * ops.arange(L_total // max_len_right + 1, dtype='int32')
        return keras_split_2D_jagged_jagged(max_seq_len=max_seq_len, values=values, offsets_left=offsets_left_non_optional, offsets_right=offsets_right_non_optional)

    return keras_split_2D_jagged_resolver(max_seq_len=max_seq_len, values=values, max_len_left=max_len_left, max_len_right=max_len_right, offsets_left=offsets_left, offsets_right=offsets_right)


def concat_2D_jagged(
    max_seq_len: int, values_left: keras.KerasTensor, values_right: keras.KerasTensor, max_len_left: Optional[int] = None, max_len_right: Optional[int] = None, offsets_left: Optional[keras.KerasTensor] = None, offsets_right: Optional[keras.KerasTensor] = None, kernel=None,
) -> keras.KerasTensor:
    """Top-level wrapper for concatenating 2D jagged tensors (used for KV cache construction)."""

    def keras_concat_2D_jagged_jagged(values_left, values_right, max_len_left, max_len_right, offsets_left, offsets_right):
        max_seq_len = max_len_left + max_len_right
        lengths_left = offsets_left[1:] - offsets_left[:-1]; lengths_right = offsets_right[1:] - offsets_right[:-1]
        padded_left = keras_jagged_to_padded_dense(values=values_left, offsets=[offsets_left], max_lengths=[max_len_left], padding_value=0.0) 
        padded_right = keras_jagged_to_padded_dense(values=values_right, offsets=[offsets_right], max_lengths=[max_len_right], padding_value=0.0) 
        concatted_dense = ops.concatenate([padded_left, padded_right], axis=1) 
        
        lengths_left_broadcast = ops.reshape(lengths_left, [-1, 1]); lengths_right_broadcast = ops.reshape(lengths_right, [-1, 1])
        mask = ops.reshape(ops.arange(max_seq_len, dtype='int32'), [1, -1])
        mask = ops.logical_or(mask < lengths_left_broadcast, ops.logical_and(mask >= max_len_left, mask < max_len_left + lengths_right_broadcast))
        return concatted_dense[ops.reshape(mask, [-1])]

    def keras_concat_2D_jagged_resolver(values_left, values_right, max_len_left, max_len_right, offsets_left, offsets_right):
        L_total = ops.shape(values_left)[0]
        offsets_left_non_optional = offsets_left
        if offsets_left is None: offsets_left_non_optional = max_len_left * ops.arange(L_total // max_len_left + 1, dtype='int32')
        offsets_right_non_optional = offsets_right
        if offsets_right is None: offsets_right_non_optional = max_len_right * ops.arange(L_total // max_len_right + 1, dtype='int32')
        
        if max_len_left is None: max_len_left_final = ops.max(offsets_left_non_optional[1:] - offsets_left_non_optional[:-1])
        else: max_len_left_final = max_len_left
        if max_len_right is None: max_len_right_final = ops.max(offsets_right_non_optional[1:] - offsets_right_non_optional[:-1])
        else: max_len_right_final = max_len_right
            
        return keras_concat_2D_jagged_jagged(values_left=values_left, values_right=values_right, max_len_left=max_len_left_final, max_len_right=max_len_right_final, offsets_left=offsets_left_non_optional, offsets_right=offsets_right_non_optional)

    return pytorch_concat_2D_jagged_resolver(values_left=values_left, values_right=values_right, max_len_left=max_len_left, max_len_right=max_len_right, offsets_left=offsets_left, offsets_right=offsets_right)
