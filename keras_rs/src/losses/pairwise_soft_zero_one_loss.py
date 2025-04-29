from keras import ops

from keras_rs.src import types
from keras_rs.src.api_export import keras_rs_export
from keras_rs.src.losses.pairwise_loss import PairwiseLoss
from keras_rs.src.losses.pairwise_loss import pairwise_loss_subclass_doc_string


@keras_rs_export("keras_rs.losses.PairwiseSoftZeroOneLoss")
class PairwiseSoftZeroOneLoss(PairwiseLoss):
    def pairwise_loss(self, pairwise_logits: types.Tensor) -> types.Tensor:
        return ops.where(
            ops.greater(pairwise_logits, ops.array(0.0)),
            ops.subtract(ops.array(1.0), ops.sigmoid(pairwise_logits)),
            ops.sigmoid(ops.negative(pairwise_logits)),
        )


formula = "loss = sum_{i} sum_{j} I(y_i > y_j) * (1 - sigmoid(s_i - s_j))"
explanation = """
    - `(1 - sigmoid(s_i - s_j))` represents the soft zero-one loss, which
      approximates the ideal zero-one loss (which would be 1 if `s_i < s_j`
      and 0 otherwise) with a smooth, differentiable function. This makes it
      suitable for gradient-based optimization.
"""
extra_args = ""
example = """
    With `compile()` API:

    ```python
    model.compile(
        loss=keras_rs.losses.PairwiseSoftZeroOneLoss(),
        ...
    )
    ```

    As a standalone function with unbatched inputs:

    >>> y_true = np.array([1.0, 0.0, 1.0, 3.0, 2.0])
    >>> y_pred = np.array([1.0, 3.0, 2.0, 4.0, 0.8])
    >>> pairwise_soft_zero_one_loss = keras_rs.losses.PairwiseSoftZeroOneLoss()
    >>> pairwise_soft_zero_one_loss(y_true=y_true, y_pred=y_pred)
    0.86103

    With batched inputs using default 'auto'/'sum_over_batch_size' reduction:

    >>> y_true = np.array([[1.0, 0.0, 1.0, 3.0], [0.0, 1.0, 2.0, 3.0]])
    >>> y_pred = np.array([[1.0, 3.0, 2.0, 4.0], [1.0, 1.8, 2.0, 3.0]])
    >>> pairwise_soft_zero_one_loss = keras_rs.losses.PairwiseSoftZeroOneLoss()
    >>> pairwise_soft_zero_one_loss(y_true=y_true, y_pred=y_pred)
    0.46202

    With masked inputs (useful for ragged inputs):

    >>> y_true = {
    ...     "labels": np.array([[1.0, 0.0, 1.0, 3.0], [0.0, 1.0, 2.0, 3.0]]),
    ...     "mask": np.array(
    ...         [[True, True, True, True], [True, True, False, False]]
    ...     ),
    ... }
    >>> y_pred = np.array([[1.0, 3.0, 2.0, 4.0], [1.0, 1.8, 2.0, 3.0]])
    >>> pairwise_soft_zero_one_loss(y_true=y_true, y_pred=y_pred)
    0.29468

    With `sample_weight`:

    >>> y_true = np.array([[1.0, 0.0, 1.0, 3.0], [0.0, 1.0, 2.0, 3.0]])
    >>> y_pred = np.array([[1.0, 3.0, 2.0, 4.0], [1.0, 1.8, 2.0, 3.0]])
    >>> sample_weight = np.array(
    ...     [[2.0, 3.0, 1.0, 1.0], [2.0, 1.0, 0.0, 0.0]]
    ... )
    >>> pairwise_soft_zero_one_loss = keras_rs.losses.PairwiseSoftZeroOneLoss()
    >>> pairwise_soft_zero_one_loss(
    ...     y_true=y_true, y_pred=y_pred, sample_weight=sample_weight
    ... )
    0.40478

    Using `'none'` reduction:

    >>> y_true = np.array([[1.0, 0.0, 1.0, 3.0], [0.0, 1.0, 2.0, 3.0]])
    >>> y_pred = np.array([[1.0, 3.0, 2.0, 4.0], [1.0, 1.8, 2.0, 3.0]])
    >>> pairwise_soft_zero_one_loss = keras_rs.losses.PairwiseSoftZeroOneLoss(
    ...     reduction="none"
    ... )
    >>> pairwise_soft_zero_one_loss(y_true=y_true, y_pred=y_pred)
    [
        [0.8807971 , 0., 0.73105854, 0.43557024],
        [0., 0.31002545, 0.7191075 , 0.61961967]
    ]
"""

PairwiseSoftZeroOneLoss.__doc__ = pairwise_loss_subclass_doc_string.format(
    loss_name="soft zero-one loss",
    formula=formula,
    explanation=explanation,
    extra_args=extra_args,
    example=example,
)
