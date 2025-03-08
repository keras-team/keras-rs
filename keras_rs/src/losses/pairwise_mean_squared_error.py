from typing import Optional

from keras import ops

from keras_rs.src import types
from keras_rs.src.api_export import keras_rs_export
from keras_rs.src.losses.pairwise_loss import PairwiseLoss
from keras_rs.src.losses.pairwise_loss import pairwise_loss_subclass_doc_string
from keras_rs.src.utils.pairwise_loss_utils import apply_pairwise_op


@keras_rs_export("keras_rs.losses.PairwiseMeanSquaredError")
class PairwiseMeanSquaredError(PairwiseLoss):
    def pairwise_loss(self, pairwise_logits: types.Tensor) -> types.Tensor:
        # Since we override `compute_unreduced_loss`, we do not need to
        # implement this method.
        pass

    def compute_unreduced_loss(
        self,
        labels: types.Tensor,
        logits: types.Tensor,
        mask: Optional[types.Tensor] = None,
    ) -> tuple[types.Tensor, types.Tensor]:
        # Override `PairwiseLoss.compute_unreduced_loss` since pairwise weights
        # for MSE are computed differently.

        batch_size, list_size = ops.shape(labels)

        # Mask all values less than 0 (since less than 0 implies invalid
        # labels).
        valid_mask = ops.greater_equal(labels, ops.cast(0.0, labels.dtype))

        if mask is not None:
            valid_mask = ops.logical_and(valid_mask, mask)

        # Compute the difference for all pairs in a list. The output is a tensor
        # with shape `(batch_size, list_size, list_size)`, where `[:, i, j]`
        # stores information for pair `(i, j)`.
        pairwise_labels_diff = apply_pairwise_op(labels, ops.subtract)
        pairwise_logits_diff = apply_pairwise_op(logits, ops.subtract)
        valid_pair = apply_pairwise_op(valid_mask, ops.logical_and)
        pairwise_mse = ops.square(pairwise_labels_diff - pairwise_logits_diff)

        # Compute weights.
        pairwise_weights = ops.ones_like(pairwise_mse)
        # Exclude self pairs.
        pairwise_weights = ops.subtract(
            pairwise_weights,
            ops.tile(ops.eye(list_size, list_size), (batch_size, 1, 1)),
        )
        # Include only valid pairs.
        pairwise_weights = ops.multiply(
            pairwise_weights, ops.cast(valid_pair, dtype=pairwise_weights.dtype)
        )

        return pairwise_mse, pairwise_weights


formula = "loss = sum_{i} sum_{j} I(y_i > y_j) * (s_i - s_j)^2"
explanation = """
      - `(s_i - s_j)^2` is the squared difference between the predicted scores
        of items `i` and `j`, which penalizes discrepancies between the
        predicted order of items relative to their true order.
"""
extra_args = ""
PairwiseMeanSquaredError.__doc__ = pairwise_loss_subclass_doc_string.format(
    formula=formula,
    explanation=explanation,
    extra_args=extra_args,
)
