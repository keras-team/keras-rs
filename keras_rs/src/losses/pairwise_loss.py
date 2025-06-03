import abc
from typing import Any

import keras
from keras import ops

from keras_rs.src import types
from keras_rs.src.losses.pairwise_loss_utils import pairwise_comparison
from keras_rs.src.metrics.utils import standardize_call_inputs_ranks


class PairwiseLoss(keras.losses.Loss, abc.ABC):
    """Base class for pairwise ranking losses.

    Pairwise loss functions are designed for ranking tasks, where the goal is to
    correctly order items within each list. Any pairwise loss function computes
    the loss value by comparing pairs of items within each list, penalizing
    cases where an item with a higher true label has a lower predicted score
    than an item with a lower true label.

    In order to implement any kind of pairwise loss, override the
    `pairwise_loss` method.
    """

    def __init__(self, temperature: float = 1.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        if temperature <= 0.0:
            raise ValueError(
                f"`temperature` should be a positive float. Received: "
                f"`temperature` = {temperature}."
            )

        self.temperature = temperature

        # TODO(abheesht): Add `lambda_weights`.

    @abc.abstractmethod
    def pairwise_loss(self, pairwise_logits: types.Tensor) -> types.Tensor:
        pass

    def compute_unreduced_loss(
        self,
        labels: types.Tensor,
        logits: types.Tensor,
        mask: types.Tensor | None = None,
    ) -> tuple[types.Tensor, types.Tensor]:
        # Mask all values less than 0 (since less than 0 implies invalid
        # labels).
        valid_mask = ops.greater_equal(labels, ops.cast(0.0, labels.dtype))

        if mask is not None:
            valid_mask = ops.logical_and(valid_mask, mask)

        pairwise_labels, pairwise_logits = pairwise_comparison(
            labels=labels,
            logits=logits,
            mask=valid_mask,
            logits_op=ops.subtract,
        )
        pairwise_logits = ops.divide(
            pairwise_logits,
            ops.cast(self.temperature, dtype=pairwise_logits.dtype),
        )

        return self.pairwise_loss(pairwise_logits), pairwise_labels

    def call(
        self,
        y_true: types.Tensor,
        y_pred: types.Tensor,
    ) -> types.Tensor:
        """Compute the pairwise loss.

        Args:
            y_true: tensor or dict. Ground truth values. If tensor, of shape
                `(list_size)` for unbatched inputs or `(batch_size, list_size)`
                for batched inputs. If an item has a label of -1, it is ignored
                in loss computation. If it is a dictionary, it should have two
                keys: `"labels"` and `"mask"`. `"mask"` can be used to ignore
                elements in loss computation, i.e., pairs will not be formed
                with those items. Note that the final mask is an `and` of the
                passed mask, and `labels >= 0`.
            y_pred: tensor. The predicted values, of shape `(list_size)` for
                unbatched inputs or `(batch_size, list_size)` for batched
                inputs. Should be of the same shape as `y_true`.

        Returns:
            The loss.
        """
        mask = None
        if isinstance(y_true, dict):
            if "labels" not in y_true:
                raise ValueError(
                    '`"labels"` should be present in `y_true`. Received: '
                    f"`y_true` = {y_true}"
                )

            mask = y_true.get("mask", None)
            y_true = y_true["labels"]

        y_true = ops.convert_to_tensor(y_true)
        y_pred = ops.convert_to_tensor(y_pred)
        if mask is not None:
            mask = ops.convert_to_tensor(mask)

        y_true, y_pred, mask, _ = standardize_call_inputs_ranks(
            y_true, y_pred, mask
        )

        losses, weights = self.compute_unreduced_loss(
            labels=y_true, logits=y_pred, mask=mask
        )
        losses = ops.multiply(losses, weights)
        losses = ops.sum(losses, axis=-1)
        return losses

    def get_config(self) -> dict[str, Any]:
        config: dict[str, Any] = super().get_config()
        config.update({"temperature": self.temperature})
        return config


pairwise_loss_subclass_doc_string = (
    "Computes pairwise {loss_name} between true labels and predicted scores."
    """
    This loss function is designed for ranking tasks, where the goal is to
    correctly order items within each list. It computes the loss by comparing
    pairs of items within each list, penalizing cases where an item with a
    higher true label has a lower predicted score than an item with a lower
    true label.

    For each list of predicted scores `s` in `y_pred` and the corresponding list
    of true labels `y` in `y_true`, the loss is computed as follows:

    ```
    {formula}
    ```

    where:

    - `y_i` and `y_j` are the true labels of items `i` and `j`, respectively.
    - `s_i` and `s_j` are the predicted scores of items `i` and `j`,
      respectively.
    - `I(y_i > y_j)` is an indicator function that equals 1 if `y_i > y_j`,
      and 0 otherwise.{explanation}
    Args:{extra_args}
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`. Supported options are
            `"sum"`, `"sum_over_batch_size"`, `"mean"`,
            `"mean_with_sample_weight"` or `None`. `"sum"` sums the loss,
            `"sum_over_batch_size"` and `"mean"` sum the loss and divide by the
            sample size, and `"mean_with_sample_weight"` sums the loss and
            divides by the sum of the sample weights. `"none"` and `None`
            perform no aggregation. Defaults to `"sum_over_batch_size"`.
        name: Optional name for the loss instance.
        dtype: The dtype of the loss's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to different value
            (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
            provided, then the `compute_dtype` will be utilized.

    Examples:
{example}"""
)
