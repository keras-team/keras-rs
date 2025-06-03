from typing import Any, Callable

from keras import ops
from keras.saving import deserialize_keras_object
from keras.saving import serialize_keras_object

from keras_rs.src import types
from keras_rs.src.api_export import keras_rs_export
from keras_rs.src.metrics.ranking_metric import RankingMetric
from keras_rs.src.metrics.ranking_metric import (
    ranking_metric_subclass_doc_string,
)
from keras_rs.src.metrics.ranking_metric import (
    ranking_metric_subclass_doc_string_post_desc,
)
from keras_rs.src.metrics.ranking_metrics_utils import compute_dcg
from keras_rs.src.metrics.ranking_metrics_utils import default_gain_fn
from keras_rs.src.metrics.ranking_metrics_utils import default_rank_discount_fn
from keras_rs.src.metrics.ranking_metrics_utils import get_list_weights
from keras_rs.src.metrics.ranking_metrics_utils import sort_by_scores
from keras_rs.src.utils.doc_string_utils import format_docstring


@keras_rs_export("keras_rs.metrics.DCG")
class DCG(RankingMetric):
    def __init__(
        self,
        k: int | None = None,
        gain_fn: Callable[[types.Tensor], types.Tensor] = default_gain_fn,
        rank_discount_fn: Callable[
            [types.Tensor], types.Tensor
        ] = default_rank_discount_fn,
        **kwargs: Any,
    ) -> None:
        super().__init__(k=k, **kwargs)

        self.gain_fn = gain_fn
        self.rank_discount_fn = rank_discount_fn

    def compute_metric(
        self,
        y_true: types.Tensor,
        y_pred: types.Tensor,
        mask: types.Tensor,
        sample_weight: types.Tensor,
    ) -> types.Tensor:
        sorted_y_true, sorted_weights = sort_by_scores(
            tensors_to_sort=[y_true, sample_weight],
            scores=y_pred,
            k=self.k,
            mask=mask,
            shuffle_ties=self.shuffle_ties,
            seed=self.seed_generator,
        )

        dcg = compute_dcg(
            y_true=sorted_y_true,
            sample_weight=sorted_weights,
            gain_fn=self.gain_fn,
            rank_discount_fn=self.rank_discount_fn,
        )

        per_list_weights = get_list_weights(
            weights=sample_weight, relevance=self.gain_fn(y_true)
        )
        # Since we have already multiplied with `sample_weight`, we need to
        # divide by `per_list_weights` so as to nullify the multiplication
        # which `keras.metrics.Mean` will do.
        per_list_dcg = ops.divide_no_nan(dcg, per_list_weights)

        return per_list_dcg, per_list_weights

    def get_config(self) -> dict[str, Any]:
        config: dict[str, Any] = super().get_config()
        config.update(
            {
                "gain_fn": serialize_keras_object(self.gain_fn),
                "rank_discount_fn": serialize_keras_object(
                    self.rank_discount_fn
                ),
            }
        )
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DCG":
        config["gain_fn"] = deserialize_keras_object(config["gain_fn"])
        config["rank_discount_fn"] = deserialize_keras_object(
            config["rank_discount_fn"]
        )
        return cls(**config)


concept_sentence = (
    "It computes the sum of the graded relevance scores of items, applying a "
    "configurable discount based on position"
)
relevance_type = (
    "graded relevance scores (non-negative numbers where higher values "
    "indicate greater relevance)"
)
score_range_interpretation = (
    "Scores are non-negative, with higher values indicating better ranking "
    "quality (highly relevant items are ranked higher). The score for a single "
    "list is not bounded or normalized, i.e., it does not lie in a range"
)

formula = """
```
DCG@k(y', w') = sum_{i=1}^{k} (gain_fn(y'_i) / rank_discount_fn(i))
```

where:
- `y'_i` is the true relevance score of the item ranked at position `i`
  (obtained by sorting `y_true` according to `y_pred`).
- `gain_fn` is the user-provided function mapping relevance `y'_i` to a
  gain value. The default function (`default_gain_fn`) is typically
  equivalent to `lambda y: 2**y - 1`.
- `rank_discount_fn` is the user-provided function mapping rank `i`
  to a discount value. The default function (`default_rank_discount_fn`)
  is typically equivalent to `lambda rank: 1 / log2(rank + 1)`.
- The final result aggregates these per-list scores."""
extra_args = """
        gain_fn: callable. Maps relevance scores (`y_true`) to gain values. The
            default implements `2**y - 1`.
        rank_discount_fn: function. Maps rank positions to discount
            values. The default (`default_rank_discount_fn`) implements
            `1 / log2(rank + 1)`."""
example = """
    >>> batch_size = 2
    >>> list_size = 5
    >>> labels = np.random.randint(0, 3, size=(batch_size, list_size))
    >>> scores = np.random.random(size=(batch_size, list_size))
    >>> metric = keras_rs.metrics.DCG()(
    ...     y_true=labels, y_pred=scores
    ... )

    Mask certain elements (can be used for uneven inputs):

    >>> batch_size = 2
    >>> list_size = 5
    >>> labels = np.random.randint(0, 3, size=(batch_size, list_size))
    >>> scores = np.random.random(size=(batch_size, list_size))
    >>> mask = np.random.randint(0, 2, size=(batch_size, list_size), dtype=bool)
    >>> metric = keras_rs.metrics.DCG()(
    ...     y_true={"labels": labels, "mask": mask}, y_pred=scores
    ... )
"""

DCG.__doc__ = format_docstring(
    ranking_metric_subclass_doc_string,
    width=80,
    metric_name="Discounted Cumulative Gain",
    metric_abbreviation="DCG",
    concept_sentence=concept_sentence,
    relevance_type=relevance_type,
    score_range_interpretation=score_range_interpretation,
    formula=formula,
) + ranking_metric_subclass_doc_string_post_desc.format(
    extra_args=extra_args, example=example
)
