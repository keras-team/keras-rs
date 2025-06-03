from keras import ops

from keras_rs.src import types
from keras_rs.src.api_export import keras_rs_export
from keras_rs.src.losses.pairwise_loss import PairwiseLoss
from keras_rs.src.losses.pairwise_loss import pairwise_loss_subclass_doc_string
from keras_rs.src.losses.pairwise_loss_utils import apply_pairwise_op


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
        mask: types.Tensor | None = None,
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
      of items `i` and `j`, which penalizes discrepancies between the predicted
      order of items relative to their true order.
"""
extra_args = ""
example = """
    With `compile()` API:

    ```python
    model.compile(
        loss=keras_rs.losses.PairwiseMeanSquaredError(),
        ...
    )
    ```

    As a standalone function with unbatched inputs:

    >>> y_true = np.array([1.0, 0.0, 1.0, 3.0, 2.0])
    >>> y_pred = np.array([1.0, 3.0, 2.0, 4.0, 0.8])
    >>> pairwise_mse = keras_rs.losses.PairwiseMeanSquaredError()
    >>> pairwise_mse(y_true=y_true, y_pred=y_pred)
    >>> 19.10400

    With batched inputs using default 'auto'/'sum_over_batch_size' reduction:

    >>> y_true = np.array([[1.0, 0.0, 1.0, 3.0], [0.0, 1.0, 2.0, 3.0]])
    >>> y_pred = np.array([[1.0, 3.0, 2.0, 4.0], [1.0, 1.8, 2.0, 3.0]])
    >>> pairwise_mse = keras_rs.losses.PairwiseMeanSquaredError()
    >>> pairwise_mse(y_true=y_true, y_pred=y_pred)
    5.57999
    
    With masked inputs (useful for ragged inputs):

    >>> y_true = {
    ...     "labels": np.array([[1.0, 0.0, 1.0, 3.0], [0.0, 1.0, 2.0, 3.0]]),
    ...     "mask": np.array(
    ...         [[True, True, True, True], [True, True, False, False]]
    ...     ),
    ... }
    >>> y_pred = np.array([[1.0, 3.0, 2.0, 4.0], [1.0, 1.8, 2.0, 3.0]])
    >>> pairwise_mse(y_true=y_true, y_pred=y_pred)
    4.76000

    With `sample_weight`:

    >>> y_true = np.array([[1.0, 0.0, 1.0, 3.0], [0.0, 1.0, 2.0, 3.0]])
    >>> y_pred = np.array([[1.0, 3.0, 2.0, 4.0], [1.0, 1.8, 2.0, 3.0]])
    >>> sample_weight = np.array(
    ...     [[2.0, 3.0, 1.0, 1.0], [2.0, 1.0, 0.0, 0.0]]
    ... )
    >>> pairwise_mse = keras_rs.losses.PairwiseMeanSquaredError()
    >>> pairwise_mse(
    ...     y_true=y_true, y_pred=y_pred, sample_weight=sample_weight
    ... )
    11.0500

    Using `'none'` reduction:

    >>> y_true = np.array([[1.0, 0.0, 1.0, 3.0], [0.0, 1.0, 2.0, 3.0]])
    >>> y_pred = np.array([[1.0, 3.0, 2.0, 4.0], [1.0, 1.8, 2.0, 3.0]])
    >>> pairwise_mse = keras_rs.losses.PairwiseMeanSquaredError(
    ...     reduction="none"
    ... )
    >>> pairwise_mse(y_true=y_true, y_pred=y_pred)
    [[11., 17.,  5.,  5.], [2.04, 1.3199998, 1.6399999, 1.6399999]]
"""

PairwiseMeanSquaredError.__doc__ = pairwise_loss_subclass_doc_string.format(
    loss_name="mean squared error",
    formula=formula,
    explanation=explanation,
    extra_args=extra_args,
    example=example,
)
