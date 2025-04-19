from typing import Callable

from keras import ops

from keras_rs.src import types


def apply_pairwise_op(
    x: types.Tensor, op: Callable[[types.Tensor, types.Tensor], types.Tensor]
) -> types.Tensor:
    return op(
        ops.expand_dims(x, axis=-1),
        ops.expand_dims(x, axis=-2),
    )


def pairwise_comparison(
    labels: types.Tensor,
    logits: types.Tensor,
    mask: types.Tensor,
    logits_op: Callable[[types.Tensor, types.Tensor], types.Tensor],
) -> tuple[types.Tensor, types.Tensor]:
    # Compute the difference for all pairs in a list. The output is a tensor
    # with shape `(batch_size, list_size, list_size)`, where `[:, i, j]` stores
    # information for pair `(i, j)`.
    pairwise_labels_diff = apply_pairwise_op(labels, ops.subtract)
    pairwise_logits = apply_pairwise_op(logits, logits_op)

    # Keep only those cases where `l_i < l_j`.
    pairwise_labels = ops.cast(
        ops.greater(pairwise_labels_diff, 0), dtype=labels.dtype
    )
    if mask is not None:
        valid_pairs = apply_pairwise_op(mask, ops.logical_and)
        pairwise_labels = ops.multiply(
            pairwise_labels, ops.cast(valid_pairs, dtype=pairwise_labels.dtype)
        )

    return pairwise_labels, pairwise_logits
