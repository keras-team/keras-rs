from typing import Any

import keras
from keras import ops

from keras_rs.src import types
from keras_rs.src.api_export import keras_rs_export
from keras_rs.src.metrics.ranking_metrics_utils import sort_by_scores
from keras_rs.src.metrics.utils import standardize_call_inputs_ranks


@keras_rs_export("keras_rs.losses.ListMLELoss")
class ListMLELoss(keras.losses.Loss):
    """Implements ListMLE (Maximum Likelihood Estimation) loss for ranking.

    ListMLE loss is a listwise ranking loss that maximizes the likelihood of
    the ground truth ranking. It works by:
    1. Sorting items by their relevance scores (labels)
    2. Computing the probability of observing this ranking given the
       predicted scores
    3. Maximizing this likelihood (minimizing negative log-likelihood)

    The loss is computed as the negative log-likelihood of the ground truth
    ranking given the predicted scores:

    ```
    loss = -sum(log(exp(s_i) / sum(exp(s_j) for j >= i)))
    ```

    where s_i is the predicted score for item i in the sorted order.

    Args:
        temperature: Temperature parameter for scaling logits. Higher values
            make the probability distribution more uniform. Defaults to 1.0.
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`. Supported options are
            `"sum"`, `"sum_over_batch_size"`, `"mean"`,
            `"mean_with_sample_weight"` or `None`. Defaults to
            `"sum_over_batch_size"`.
        name: Optional name for the loss instance.
        dtype: The dtype of the loss's computations. Defaults to `None`.

    Examples:
        ```python
        # Basic usage
        loss_fn = ListMLELoss()

        # With temperature scaling
        loss_fn = ListMLELoss(temperature=0.5)

        # Example with synthetic data
        y_true = [[3, 2, 1, 0]]  # Relevance scores
        y_pred = [[0.8, 0.6, 0.4, 0.2]]  # Predicted scores
        loss = loss_fn(y_true, y_pred)
        ```
    """

    def __init__(self, temperature: float = 1.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        if temperature <= 0.0:
            raise ValueError(
                f"`temperature` should be a positive float. Received: "
                f"`temperature` = {temperature}."
            )

        self.temperature = temperature
        self._epsilon = 1e-10

    def compute_unreduced_loss(
        self,
        labels: types.Tensor,
        logits: types.Tensor,
        mask: types.Tensor | None = None,
    ) -> tuple[types.Tensor, types.Tensor]:
        """Compute the unreduced ListMLE loss.

        Args:
            labels: Ground truth relevance scores of
                    shape [batch_size,list_size].
            logits: Predicted scores of shape [batch_size, list_size].
            mask: Optional mask of shape [batch_size, list_size].

        Returns:
            Tuple of (losses, weights) where losses has shape [batch_size, 1]
            and weights has the same shape.
        """

        valid_mask = ops.greater_equal(labels, ops.cast(0.0, labels.dtype))

        if mask is not None:
            valid_mask = ops.logical_and(
                valid_mask, ops.cast(mask, dtype="bool")
            )

        num_valid_items = ops.sum(
            ops.cast(valid_mask, dtype=labels.dtype), axis=1, keepdims=True
        )

        batch_has_valid_items = ops.greater(num_valid_items, 0.0)

        labels_for_sorting = ops.where(
            valid_mask, labels, ops.full_like(labels, -1e9)
        )
        logits_masked = ops.where(
            valid_mask, logits, ops.full_like(logits, -1e9)
        )

        sorted_logits, sorted_valid_mask = sort_by_scores(
            tensors_to_sort=[logits_masked, valid_mask],
            scores=labels_for_sorting,
            mask=None,
            shuffle_ties=False,
            seed=None,
        )
        sorted_logits = ops.divide(
            sorted_logits, ops.cast(self.temperature, dtype=sorted_logits.dtype)
        )

        valid_logits_for_max = ops.where(
            sorted_valid_mask, sorted_logits, ops.full_like(sorted_logits, -1e9)
        )
        raw_max = ops.max(valid_logits_for_max, axis=1, keepdims=True)
        raw_max = ops.where(
            batch_has_valid_items, raw_max, ops.zeros_like(raw_max)
        )
        sorted_logits = ops.subtract(sorted_logits, raw_max)

        # Set invalid positions to very negative BEFORE exp
        sorted_logits = ops.where(
            sorted_valid_mask, sorted_logits, ops.full_like(sorted_logits, -1e9)
        )
        exp_logits = ops.exp(sorted_logits)

        reversed_exp = ops.flip(exp_logits, axis=1)
        reversed_cumsum = ops.cumsum(reversed_exp, axis=1)
        cumsum_from_right = ops.flip(reversed_cumsum, axis=1)

        log_normalizers = ops.log(cumsum_from_right + self._epsilon)
        log_probs = ops.subtract(sorted_logits, log_normalizers)

        log_probs = ops.where(
            sorted_valid_mask, log_probs, ops.zeros_like(log_probs)
        )

        negative_log_likelihood = ops.negative(
            ops.sum(log_probs, axis=1, keepdims=True)
        )

        negative_log_likelihood = ops.where(
            batch_has_valid_items,
            negative_log_likelihood,
            ops.zeros_like(negative_log_likelihood),
        )

        weights = ops.ones_like(negative_log_likelihood)

        return negative_log_likelihood, weights

    def call(
        self,
        y_true: types.Tensor,
        y_pred: types.Tensor,
    ) -> types.Tensor:
        """Compute the ListMLE loss.

        Args:
            y_true: tensor or dict. Ground truth values. If tensor, of shape
                `(list_size)` for unbatched inputs or `(batch_size, list_size)`
                for batched inputs. If an item has a label of -1, it is ignored
                in loss computation. If it is a dictionary, it should have two
                keys: `"labels"` and `"mask"`. `"mask"` can be used to ignore
                elements in loss computation.
            y_pred: tensor. The predicted values, of shape `(list_size)` for
                unbatched inputs or `(batch_size, list_size)` for batched
                inputs. Should be of the same shape as `y_true`.

        Returns:
            The loss tensor of shape [batch_size].
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
        losses = ops.squeeze(losses, axis=-1)
        return losses

    # getting config
    def get_config(self) -> dict[str, Any]:
        config: dict[str, Any] = super().get_config()
        config.update({"temperature": self.temperature})
        return config
