from typing import Optional

from keras import ops

from keras_rs.src import types
from keras_rs.src.utils.keras_utils import check_shapes_compatible


def apply_pairwise_op(x: types.Tensor, op: ops) -> types.Tensor:
    return op(
        ops.expand_dims(x, axis=-1),
        ops.expand_dims(x, axis=-2),
    )


def pairwise_comparison(
    labels: types.Tensor,
    logits: types.Tensor,
    mask: types.Tensor,
    logits_op: ops,
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


def process_loss_call_inputs(
    y_true: types.Tensor,
    y_pred: types.Tensor,
    sample_weight: Optional[types.Tensor] = None,
) -> tuple[types.Tensor, types.Tensor, types.Tensor]:
    """
    This utility function does three things:

    - Checks that `y_true`, `y_pred` are of rank 1 or 2. `sample_weight` can be
      of ranks 0, 1, 2.
    - Check that `y_true` and `y_pred` have the same shape.
    - Add batch dimension if rank = 1.
    """

    def get_shape(
        x: types.Tensor, convert_to_tensor: bool = False
    ) -> tuple[types.Tensor, types.TensorShape, int]:
        if convert_to_tensor and isinstance(x, (list, float, int)):
            x = ops.array(x)
        shape = ops.shape(x)
        rank = len(shape)
        return x, shape, rank

    # These are typecast to tensors in `keras.losses.Loss.__call__()`. So,
    # not need to convert them to tensors.
    y_true, y_true_shape, y_true_rank = get_shape(y_true)
    y_pred, y_pred_shape, y_pred_rank = get_shape(y_pred)
    # `keras.losses.Loss.__call__()` does not convert `sample_weight` to tensor.
    if sample_weight is not None:
        sample_weight, sample_weight_shape, sample_weight_rank = get_shape(
            sample_weight, convert_to_tensor=True
        )

    # Check ranks and shapes.
    def check_rank(
        x_rank: int,
        allowed_ranks: tuple[int, ...] = (1, 2),
        tensor_name: Optional[str] = None,
    ) -> None:
        if x_rank not in allowed_ranks:
            raise ValueError(
                f"`{tensor_name}` should have a rank from `{allowed_ranks}`."
                f"Received: `{x_rank}`."
            )

    check_rank(y_true_rank, tensor_name="y_true")
    check_rank(y_pred_rank, tensor_name="y_pred")
    if not check_shapes_compatible(y_true_shape, y_pred_shape):
        raise ValueError(
            "`y_true` and `y_pred` should have the same shape. Received: "
            f"`y_true.shape` = {y_true_shape}, `y_pred.shape` = {y_pred_shape}."
        )
    if sample_weight is not None:
        check_rank(
            sample_weight_rank,
            allowed_ranks=tuple(range(y_true_rank + 1)),
            tensor_name="sample_weight",
        )

    if y_true_rank == 1:
        # Do not need to modify sample_weight. `keras.losses.Loss` takes care of
        # it.
        y_true = ops.expand_dims(y_true, axis=0)
        y_pred = ops.expand_dims(y_pred, axis=0)

    return y_true, y_pred, sample_weight
