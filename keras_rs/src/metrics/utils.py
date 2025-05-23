from keras import ops

from keras_rs.src import types
from keras_rs.src.utils.keras_utils import check_rank
from keras_rs.src.utils.keras_utils import check_shapes_compatible


def standardize_call_inputs_ranks(
    y_true: types.Tensor,
    y_pred: types.Tensor,
    mask: types.Tensor | None = None,
    check_y_true_rank: bool = True,
) -> tuple[types.Tensor, types.Tensor, types.Tensor | None, bool]:
    """
    Utility function for processing inputs for losses and metrics.

    This utility function does three things:

    - Checks that `y_true`, `y_pred` are of rank 1 or 2;
    - Checks that `y_true`, `y_pred`, `mask` have the same shape;
    - Adds batch dimension if rank = 1.

    Args:
        y_true: tensor. Ground truth values.
        y_pred: tensor. The predicted values.
        mask: tensor. Boolean mask for `y_true`.
        check_y_true_rank: bool. Whether to check the rank of `y_true`.

    Returns:
        Tuple of processed `y_true`, `y_pred`, `mask`, and `batched`. `batched`
        is a bool indicating if the inputs are batched.
    """

    y_true_shape = ops.shape(y_true)
    y_true_rank = len(y_true_shape)
    y_pred_shape = ops.shape(y_pred)
    y_pred_rank = len(y_pred_shape)
    if mask is not None:
        mask_shape = ops.shape(mask)
        mask_rank = len(mask_shape)

    if check_y_true_rank:
        check_rank(y_true_rank, allowed_ranks=(1, 2), tensor_name="y_true")
    check_rank(y_pred_rank, allowed_ranks=(1, 2), tensor_name="y_pred")
    if mask is not None:
        check_rank(mask_rank, allowed_ranks=(1, 2), tensor_name="mask")
    if not check_shapes_compatible(y_true_shape, y_pred_shape):
        raise ValueError(
            "`y_true` and `y_pred` should have the same shape. Received: "
            f"`y_true.shape` = {y_true_shape}, `y_pred.shape` = {y_pred_shape}."
        )
    if mask is not None and not check_shapes_compatible(
        y_true_shape, mask_shape
    ):
        raise ValueError(
            "`y_true['labels']` and `y_true['mask']` should have the same "
            f"shape. Received: `y_true['labels'].shape` = {y_true_shape}, "
            f"`y_true['mask'].shape` = {mask_shape}."
        )

    batched = True
    if y_true_rank == 1:
        batched = False

        y_true = ops.expand_dims(y_true, axis=0)
        y_pred = ops.expand_dims(y_pred, axis=0)
        if mask is not None:
            mask = ops.expand_dims(mask, axis=0)

    return y_true, y_pred, mask, batched
