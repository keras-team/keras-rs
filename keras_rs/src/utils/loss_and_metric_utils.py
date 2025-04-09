from typing import Optional

from keras import ops

from keras_rs.src import types
from keras_rs.src.utils.keras_utils import check_rank
from keras_rs.src.utils.keras_utils import check_shapes_compatible


def process_inputs(
    y_true: types.Tensor,
    y_pred: types.Tensor,
    mask: Optional[types.Tensor] = None,
    check_y_true_rank: bool = True,
) -> tuple[types.Tensor, types.Tensor, Optional[types.Tensor]]:
    """
    Utility function for processing inputs for pairwise losses.

    This utility function does three things:

    - Checks that `y_true`, `y_pred` are of rank 1 or 2;
    - Checks that `y_true`, `y_pred`, `mask` have the same shape;
    - Adds batch dimension if rank = 1.
    """

    y_true_shape = ops.shape(y_true)
    y_true_rank = len(y_true_shape)
    y_pred_shape = ops.shape(y_pred)
    y_pred_rank = len(y_pred_shape)
    if mask is not None:
        mask_shape = ops.shape(mask)
        mask_rank = len(mask_shape)

    if check_y_true_rank:
        check_rank(y_true_rank, tensor_name="y_true")
    check_rank(y_pred_rank, tensor_name="y_pred")
    if mask is not None:
        check_rank(mask_rank, tensor_name="mask")
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

    if y_true_rank == 1:
        y_true = ops.expand_dims(y_true, axis=0)
        y_pred = ops.expand_dims(y_pred, axis=0)
        if mask is not None:
            mask = ops.expand_dims(mask, axis=0)

    return y_true, y_pred, mask
