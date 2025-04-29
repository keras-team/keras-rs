from keras import ops

from keras_rs.src import types
from keras_rs.src.api_export import keras_rs_export
from keras_rs.src.losses.pairwise_loss import PairwiseLoss
from keras_rs.src.losses.pairwise_loss import pairwise_loss_subclass_doc_string


@keras_rs_export("keras_rs.losses.PairwiseHingeLoss")
class PairwiseHingeLoss(PairwiseLoss):
    def pairwise_loss(self, pairwise_logits: types.Tensor) -> types.Tensor:
        return ops.relu(ops.subtract(ops.array(1), pairwise_logits))


formula = "loss = sum_{i} sum_{j} I(y_i > y_j) * max(0, 1 - (s_i - s_j))"
explanation = """
    - `max(0, 1 - (s_i - s_j))` is the hinge loss, which penalizes cases where
      the score difference `s_i - s_j` is not sufficiently large when
      `y_i > y_j`.
"""
extra_args = ""
example = """
    With `compile()` API:

    ```python
    model.compile(
        loss=keras_rs.losses.PairwiseHingeLoss(),
        ...
    )
    ```

    As a standalone function with unbatched inputs:

    >>> y_true = np.array([1.0, 0.0, 1.0, 3.0, 2.0])
    >>> y_pred = np.array([1.0, 3.0, 2.0, 4.0, 0.8])
    >>> pairwise_hinge_loss = keras_rs.losses.PairwiseHingeLoss()
    >>> pairwise_hinge_loss(y_true=y_true, y_pred=y_pred)
    2.32000

    With batched inputs using default 'auto'/'sum_over_batch_size' reduction:

    >>> y_true = np.array([[1.0, 0.0, 1.0, 3.0], [0.0, 1.0, 2.0, 3.0]])
    >>> y_pred = np.array([[1.0, 3.0, 2.0, 4.0], [1.0, 1.8, 2.0, 3.0]])
    >>> pairwise_hinge_loss = keras_rs.losses.PairwiseHingeLoss()
    >>> pairwise_hinge_loss(y_true=y_true, y_pred=y_pred)
    0.75

    With masked inputs (useful for ragged inputs):

    >>> y_true = {
    ...     "labels": np.array([[1.0, 0.0, 1.0, 3.0], [0.0, 1.0, 2.0, 3.0]]),
    ...     "mask": np.array(
    ...         [[True, True, True, True], [True, True, False, False]]
    ...     ),
    ... }
    >>> y_pred = np.array([[1.0, 3.0, 2.0, 4.0], [1.0, 1.8, 2.0, 3.0]])
    >>> pairwise_hinge_loss(y_true=y_true, y_pred=y_pred)
    0.64999

    With `sample_weight`:

    >>> y_true = np.array([[1.0, 0.0, 1.0, 3.0], [0.0, 1.0, 2.0, 3.0]])
    >>> y_pred = np.array([[1.0, 3.0, 2.0, 4.0], [1.0, 1.8, 2.0, 3.0]])
    >>> sample_weight = np.array(
    ...     [[2.0, 3.0, 1.0, 1.0], [2.0, 1.0, 0.0, 0.0]]
    ... )
    >>> pairwise_hinge_loss = keras_rs.losses.PairwiseHingeLoss()
    >>> pairwise_hinge_loss(
    ...     y_true=y_true, y_pred=y_pred, sample_weight=sample_weight
    ... )
    1.02499

    Using `'none'` reduction:

    >>> y_true = np.array([[1.0, 0.0, 1.0, 3.0], [0.0, 1.0, 2.0, 3.0]])
    >>> y_pred = np.array([[1.0, 3.0, 2.0, 4.0], [1.0, 1.8, 2.0, 3.0]])
    >>> pairwise_hinge_loss = keras_rs.losses.PairwiseHingeLoss(
    ...     reduction="none"
    ... )
    >>> pairwise_hinge_loss(y_true=y_true, y_pred=y_pred)
    [[3. , 0. , 2. , 0.], [0., 0.20000005, 0.79999995, 0.]]
"""

PairwiseHingeLoss.__doc__ = pairwise_loss_subclass_doc_string.format(
    loss_name="hinge loss",
    formula=formula,
    explanation=explanation,
    extra_args=extra_args,
    example=example,
)
