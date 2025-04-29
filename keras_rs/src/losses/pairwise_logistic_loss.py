from keras import ops

from keras_rs.src import types
from keras_rs.src.api_export import keras_rs_export
from keras_rs.src.losses.pairwise_loss import PairwiseLoss
from keras_rs.src.losses.pairwise_loss import pairwise_loss_subclass_doc_string


@keras_rs_export("keras_rs.losses.PairwiseLogisticLoss")
class PairwiseLogisticLoss(PairwiseLoss):
    def pairwise_loss(self, pairwise_logits: types.Tensor) -> types.Tensor:
        return ops.add(
            ops.relu(ops.negative(pairwise_logits)),
            ops.log(
                ops.add(
                    ops.array(1),
                    ops.exp(ops.negative(ops.abs(pairwise_logits))),
                )
            ),
        )


formula = "loss = sum_{i} sum_{j} I(y_i > y_j) * log(1 + exp(-(s_i - s_j)))"
explanation = """
    - `log(1 + exp(-(s_i - s_j)))` is the logistic loss, which penalizes
      cases where the score difference `s_i - s_j` is not sufficiently large
      when `y_i > y_j`. This function provides a smooth approximation of the
      ideal step function, making it suitable for gradient-based optimization.
"""
extra_args = ""
example = """
    With `compile()` API:

    ```python
    model.compile(
        loss=keras_rs.losses.PairwiseLogisticLoss(),
        ...
    )
    ```

    As a standalone function with unbatched inputs:

    >>> y_true = np.array([1.0, 0.0, 1.0, 3.0, 2.0])
    >>> y_pred = np.array([1.0, 3.0, 2.0, 4.0, 0.8])
    >>> pairwise_logistic_loss = keras_rs.losses.PairwiseLogisticLoss()
    >>> pairwise_logistic_loss(y_true=y_true, y_pred=y_pred)
    >>> 1.70708

    With batched inputs using default 'auto'/'sum_over_batch_size' reduction:

    >>> y_true = np.array([[1.0, 0.0, 1.0, 3.0], [0.0, 1.0, 2.0, 3.0]])
    >>> y_pred = np.array([[1.0, 3.0, 2.0, 4.0], [1.0, 1.8, 2.0, 3.0]])
    >>> pairwise_logistic_loss = keras_rs.losses.PairwiseLogisticLoss()
    >>> pairwise_logistic_loss(y_true=y_true, y_pred=y_pred)
    0.73936

    With masked inputs (useful for ragged inputs):

    >>> y_true = {
    ...     "labels": np.array([[1.0, 0.0, 1.0, 3.0], [0.0, 1.0, 2.0, 3.0]]),
    ...     "mask": np.array(
    ...         [[True, True, True, True], [True, True, False, False]]
    ...     ),
    ... }
    >>> y_pred = np.array([[1.0, 3.0, 2.0, 4.0], [1.0, 1.8, 2.0, 3.0]])
    >>> pairwise_logistic_loss(y_true=y_true, y_pred=y_pred)
    0.53751

    With `sample_weight`:

    >>> y_true = np.array([[1.0, 0.0, 1.0, 3.0], [0.0, 1.0, 2.0, 3.0]])
    >>> y_pred = np.array([[1.0, 3.0, 2.0, 4.0], [1.0, 1.8, 2.0, 3.0]])
    >>> sample_weight = np.array(
    ...     [[2.0, 3.0, 1.0, 1.0], [2.0, 1.0, 0.0, 0.0]]
    ... )
    >>> pairwise_logistic_loss = keras_rs.losses.PairwiseLogisticLoss()
    >>> pairwise_logistic_loss(
    ...     y_true=y_true, y_pred=y_pred, sample_weight=sample_weight
    ... )
    >>> 0.80337

    Using `'none'` reduction:

    >>> y_true = np.array([[1.0, 0.0, 1.0, 3.0], [0.0, 1.0, 2.0, 3.0]])
    >>> y_pred = np.array([[1.0, 3.0, 2.0, 4.0], [1.0, 1.8, 2.0, 3.0]])
    >>> pairwise_logistic_loss = keras_rs.losses.PairwiseLogisticLoss(
    ...     reduction="none"
    ... )
    >>> pairwise_logistic_loss(y_true=y_true, y_pred=y_pred)
    [[2.126928, 0., 1.3132616, 0.48877698], [0., 0.20000005, 0.79999995, 0.]]
"""

PairwiseLogisticLoss.__doc__ = pairwise_loss_subclass_doc_string.format(
    loss_name="logistic loss",
    formula=formula,
    explanation=explanation,
    extra_args=extra_args,
    example=example,
)
