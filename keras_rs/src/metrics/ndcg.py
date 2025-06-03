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


@keras_rs_export("keras_rs.metrics.NDCG")
class NDCG(RankingMetric):
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

        weighted_gains = ops.multiply(
            sample_weight,
            self.gain_fn(y_true),
        )
        ideal_sorted_y_true, ideal_sorted_weights = sort_by_scores(
            tensors_to_sort=[y_true, sample_weight],
            scores=weighted_gains,
            k=self.k,
            mask=mask,
            shuffle_ties=self.shuffle_ties,
            seed=self.seed_generator,
        )
        ideal_dcg = compute_dcg(
            y_true=ideal_sorted_y_true,
            sample_weight=ideal_sorted_weights,
            gain_fn=self.gain_fn,
            rank_discount_fn=self.rank_discount_fn,
        )
        per_list_ndcg = ops.divide_no_nan(dcg, ideal_dcg)

        per_list_weights = get_list_weights(
            weights=sample_weight, relevance=self.gain_fn(y_true)
        )

        return per_list_ndcg, per_list_weights

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
    def from_config(cls, config: dict[str, Any]) -> "NDCG":
        config["gain_fn"] = deserialize_keras_object(config["gain_fn"])
        config["rank_discount_fn"] = deserialize_keras_object(
            config["rank_discount_fn"]
        )
        return cls(**config)


concept_sentence = (
    "It normalizes the Discounted Cumulative Gain (DCG) with the Ideal "
    "Discounted Cumulative Gain (IDCG) for each list"
)
relevance_type = (
    "graded relevance scores (non-negative numbers where higher values "
    "indicate greater relevance)"
)
score_range_interpretation = (
    "A normalized score (between 0 and 1) is returned. A score of 1 "
    "represents the perfect ranking according to true relevance (within the "
    "top-k), while 0 typically represents a ranking with no relevant items. "
    "Higher scores indicate better ranking quality relative to the best "
    "possible ranking"
)

formula = """
```
nDCG@k = DCG@k / IDCG@k
```

where DCG@k is calculated based on the predicted ranking (`y_pred`):

```
DCG@k(y') = sum_{i=1}^{k} (gain_fn(y'_i) / rank_discount_fn(i))
```

And IDCG@k is the Ideal DCG, calculated using the same formula but on items
sorted perfectly by their *true relevance* (`y_true`):

```
IDCG@k(y'') = sum_{i=1}^{k} (gain_fn(y''_i) / rank_discount_fn(i))
```

where:
- `y'_i`: True relevance of the item at rank `i` in the ranking induced by
  `y_pred`.
- `y''_i` True relevance of the item at rank `i` in the *ideal* ranking (sorted
  by `y_true` descending).
- `gain_fn` is the user-provided function mapping relevance to gain. The default
  function (`default_gain_fn`) is typically equivalent to `lambda y: 2**y - 1`.
- `rank_discount_fn` is the user-provided function mapping rank `i` (1-based) to
  a discount value. The default function (`default_rank_discount_fn`) is
  typically equivalent to `lambda rank: 1 / log2(rank + 1)`.
- If IDCG@k is 0 (e.g., no relevant items), nDCG@k is defined as 0.
- The final result often aggregates these per-list nDCG scores, potentially
  involving normalization by list-specific weights, to produce a weighted
  average."""
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
    >>> metric = keras_rs.metrics.NDCG()(
    ...     y_true=labels, y_pred=scores
    ... )

    Mask certain elements (can be used for uneven inputs):

    >>> batch_size = 2
    >>> list_size = 5
    >>> labels = np.random.randint(0, 3, size=(batch_size, list_size))
    >>> scores = np.random.random(size=(batch_size, list_size))
    >>> mask = np.random.randint(0, 2, size=(batch_size, list_size), dtype=bool)
    >>> metric = keras_rs.metrics.NDCG()(
    ...     y_true={"labels": labels, "mask": mask}, y_pred=scores
    ... )
"""

NDCG.__doc__ = format_docstring(
    ranking_metric_subclass_doc_string,
    width=80,
    metric_name="Normalised Discounted Cumulative Gain",
    metric_abbreviation="nDCG",
    concept_sentence=concept_sentence,
    relevance_type=relevance_type,
    score_range_interpretation=score_range_interpretation,
    formula=formula,
    extra_args=extra_args,
) + ranking_metric_subclass_doc_string_post_desc.format(
    extra_args=extra_args, example=example
)
