import abc
from typing import Any

import keras
from keras import ops

from keras_rs.src import types
from keras_rs.src.metrics.utils import standardize_call_inputs_ranks
from keras_rs.src.utils.keras_utils import check_rank
from keras_rs.src.utils.keras_utils import check_shapes_compatible


class RankingMetric(keras.metrics.Mean, abc.ABC):
    """Base class for ranking evaluation metrics (e.g., MAP, MRR, DCG, nDCG).

    Ranking metrics are used to evaluate the quality of ranked lists produced
    by a ranking model. The primary goal in ranking tasks is to order items
    according to their relevance or utility for a given query or context.
    These metrics provide quantitative measures of how well a model achieves
    this goal, typically by comparing the predicted order of items against the
    ground truth relevance judgments for each list.

    To define your own ranking metric, subclass this class and implement the
    `compute_metric` method.

    Args:
        k: int. The number of top-ranked items to consider (the 'k' in 'top-k').
            Must be a positive integer.
        shuffle_ties: bool. Whether to randomly shuffle scores before sorting.
            This is done to break ties. Defaults to `True`.
        seed: int. Random seed used for shuffling.
        name: Optional name for the loss instance.
        dtype: The dtype of the metric's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to different value
            (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
            provided, then the `compute_dtype` will be utilized.
    """

    def __init__(
        self,
        k: int | None = None,
        shuffle_ties: bool = True,
        seed: int | keras.random.SeedGenerator | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if k is not None and (not isinstance(k, int) or k < 1):
            raise ValueError(
                f"`k` should be a positive integer. Received: `k` = {k}."
            )

        self.k = k
        self.shuffle_ties = shuffle_ties
        self.seed = seed

        # Define `SeedGenerator`. JAX doesn't work, otherwise.
        self.seed_generator = keras.random.SeedGenerator(seed)

    @abc.abstractmethod
    def compute_metric(
        self,
        y_true: types.Tensor,
        y_pred: types.Tensor,
        mask: types.Tensor,
        sample_weight: types.Tensor,
    ) -> types.Tensor:
        """Abstract method, should be implemented by subclasses."""
        pass

    def update_state(
        self,
        y_true: types.Tensor,
        y_pred: types.Tensor,
        sample_weight: types.Tensor | None = None,
    ) -> None:
        """
        Accumulates statistics for the ranking metric.

        Args:
            y_true: tensor or dict. Ground truth values. If tensor, of shape
                `(list_size)` for unbatched inputs or `(batch_size, list_size)`
                for batched inputs. If an item has a label of -1, it is ignored
                in loss computation. If it is a dictionary, it should have two
                keys: `"labels"` and `"mask"`. `"mask"` can be used to ignore
                elements in metric computation, i.e., pairs will not be formed
                with those items. Note that the final mask is an `and` of the
                passed mask, `labels >= 0`, and `sample_weight > 0`.
            y_pred: tensor. The predicted values, of shape `(list_size)` for
                unbatched inputs or `(batch_size, list_size)` for batched
                inputs. Should be of the same shape as `y_true`.
            sample_weight: float/tensor. Can be float value, or tensor of
                shape `(list_size)` or `(batch_size, list_size)`. Defaults to
                `None`.
        """
        # === Process `y_true`, if dict ===
        passed_mask = None
        if isinstance(y_true, dict):
            if "labels" not in y_true:
                raise ValueError(
                    '`"labels"` should be present in `y_true`. Received: '
                    f"`y_true` = {y_true}"
                )

            passed_mask = y_true.get("mask", None)
            y_true = y_true["labels"]

        # === Convert to tensors, if list ===
        # TODO (abheesht): Figure out if we need to cast tensors to
        # `self.dtype`.
        y_true = ops.convert_to_tensor(y_true)
        y_pred = ops.convert_to_tensor(y_pred)
        if sample_weight is not None:
            sample_weight = ops.convert_to_tensor(sample_weight)
        if passed_mask is not None:
            passed_mask = ops.convert_to_tensor(passed_mask)

        # Cast to the correct dtype.
        y_true = ops.cast(y_true, dtype=self.dtype)
        y_pred = ops.cast(y_pred, dtype=self.dtype)
        if sample_weight is not None:
            sample_weight = ops.cast(sample_weight, dtype=self.dtype)

        # === Process `sample_weight` ===
        if sample_weight is None:
            sample_weight = ops.cast(1, dtype=y_pred.dtype)

        y_true_shape = ops.shape(y_true)
        y_true_rank = len(y_true_shape)
        sample_weight_shape = ops.shape(sample_weight)
        sample_weight_rank = len(sample_weight_shape)

        # Check `y_true_rank` first. Can be 1 for unbatched inputs, 2 for
        # batched.
        check_rank(y_true_rank, allowed_ranks=(1, 2), tensor_name="y_true")

        # Check `sample_weight` rank. Should be between 0 and `y_true_rank`.
        check_rank(
            sample_weight_rank,
            allowed_ranks=tuple(range(y_true_rank + 1)),
            tensor_name="sample_weight",
        )

        if y_true_rank == 2:
            # If `sample_weight` rank is 1, it should be of shape
            # `(batch_size,)`. Otherwise, it should be of shape
            # `(batch_size, list_size)`.
            if sample_weight_rank == 1:
                check_shapes_compatible(sample_weight_shape, (y_true_shape[0],))
                # Uprank this, so that we get per-list weights here.
                sample_weight = ops.expand_dims(sample_weight, axis=1)
            elif sample_weight_rank == 2:
                check_shapes_compatible(sample_weight_shape, y_true_shape)

        # Reshape `sample_weight` to the shape of `y_true`.
        sample_weight = ops.multiply(ops.ones_like(y_true), sample_weight)

        # Mask all values less than 0 (since less than 0 implies invalid
        # labels).
        valid_mask = ops.greater_equal(y_true, ops.cast(0, y_true.dtype))
        if passed_mask is not None:
            valid_mask = ops.logical_and(valid_mask, passed_mask)

        # === Process inputs - shape checking, upranking, etc. ===
        y_true, y_pred, valid_mask, batched = standardize_call_inputs_ranks(
            y_true=y_true,
            y_pred=y_pred,
            mask=valid_mask,
            check_y_true_rank=False,
        )

        # Uprank sample_weight if unbatched.
        if not batched:
            sample_weight = ops.expand_dims(sample_weight, axis=0)

        # Get `mask` from `sample_weight`.
        sample_weight_mask = ops.greater(
            sample_weight, ops.cast(0, dtype=sample_weight.dtype)
        )
        mask = ops.logical_and(valid_mask, sample_weight_mask)

        # === Update "invalid" `y_true`, `y_pred` entries based on mask ===

        # `y_true`: assign 0 for invalid entries
        y_true = ops.where(mask, y_true, ops.zeros_like(y_true))
        # `y_pred`: assign a value slightly smaller than the smallest value
        # so that it gets sorted last.
        y_pred = ops.where(
            mask,
            y_pred,
            -keras.config.epsilon() * ops.ones_like(y_pred)
            + ops.amin(y_pred, axis=1, keepdims=True),
        )
        sample_weight = ops.where(
            mask, sample_weight, ops.cast(0, sample_weight.dtype)
        )

        # === Actual computation ===
        per_list_metric_values, per_list_metric_weights = self.compute_metric(
            y_true=y_true, y_pred=y_pred, mask=mask, sample_weight=sample_weight
        )

        # Chain to `super()` to get mean metric.
        # TODO(abheesht): Figure out if we want to return unaggregated metric
        # values too of shape `(batch_size,)` from `result()`.
        super().update_state(
            per_list_metric_values, sample_weight=per_list_metric_weights
        )

    def get_config(self) -> dict[str, Any]:
        config: dict[str, Any] = super().get_config()
        config.update(
            {"k": self.k, "shuffle_ties": self.shuffle_ties, "seed": self.seed}
        )
        return config


ranking_metric_subclass_doc_string = """
Computes {metric_name} ({metric_abbreviation}).

This metric evaluates ranking quality. {concept_sentence}. The metric processes
true relevance labels in `y_true` ({relevance_type}) against predicted scores in
`y_pred`. The scores in `y_pred` are used to determine the rank order of items,
by sorting in descending order. {score_range_interpretation}.

For each list of predicted scores `s` in `y_pred` and the corresponding list
of true labels `y` in `y_true`, the per-query {metric_abbreviation} score is
calculated as follows:
{formula}

The final {metric_abbreviation} score reported is typically the weighted
average of these per-query scores across all queries/lists in the dataset.

Note: `sample_weight` is handled differently for ranking metrics. For
batched inputs, `sample_weight` can be scalar, 1D, 2D. The scalar case and
1D case (list-wise weights) are straightforward. The 2D case (item-wise
weights) is different, in the sense that the sample weights are aggregated
to get 1D weights. For more details, refer to
`keras_rs.src.metrics.ranking_metrics_utils.get_list_weights`.
"""

ranking_metric_subclass_doc_string_post_desc = """

    Args:{extra_args}
        k: int. The number of top-ranked items to consider (the 'k' in 'top-k').
            Must be a positive integer.
        shuffle_ties: bool. Whether to randomly shuffle scores before sorting.
            This is done to break ties. Defaults to `True`.
        seed: int. Random seed used for shuffling.
        name: Optional name for the loss instance.
        dtype: The dtype of the metric's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to different value
            (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
            provided, then the `compute_dtype` will be utilized.

    Example:
{example}
"""
