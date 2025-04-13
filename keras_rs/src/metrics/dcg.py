from typing import Any, Callable, Optional

from keras import ops

from keras_rs.src import types
from keras_rs.src.api_export import keras_rs_export
from keras_rs.src.metrics.ranking_metric import RankingMetric
from keras_rs.src.metrics.ranking_metric import (
    ranking_metric_subclass_doc_string,
)
from keras_rs.src.utils.ranking_metrics_utils import compute_dcg
from keras_rs.src.utils.ranking_metrics_utils import default_gain_fn
from keras_rs.src.utils.ranking_metrics_utils import default_rank_discount_fn
from keras_rs.src.utils.ranking_metrics_utils import get_list_weights
from keras_rs.src.utils.ranking_metrics_utils import sort_by_scores


@keras_rs_export("keras_rs.metrics.DCG")
class DCG(RankingMetric):
    def __init__(
        self,
        k: Optional[int] = None,
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
        per_list_dcg = ops.divide_no_nan(dcg, per_list_weights)

        return per_list_dcg, per_list_weights


core_concept_sentence = (
    "a measure of ranking quality that sums the graded relevance scores of "
    "items, applying a configurable discount based on position"
)
relevance_type_description = (
    "graded relevance scores (non-negative numbers where higher values "
    "indicate greater relevance)"
)
score_range_interpretation = (
    "Returns a weighted average score per ranked list. Scores are "
    "non-negative, with higher values indicating better ranking quality "
    "(highly relevant items are ranked higher). The score for a single list "
    "is not inherently normalized between 0 and 1, and its maximum depends on "
    "the specific relevance scores, weights, and list length (or cutoff `k`)."
)

formula = """
    ```
    DCG@k(y', w') = sum_{i=1}^{k} (gain_fn(y'_i) / rank_discount_fn(i))
    ```

    where:
        - `k` is the rank position cutoff (determined by the `k` parameter or
          list size).
        - The sum is over the top `k` ranks `i` (from 1 to `k`).
        - `y'_i` is the true relevance score of the item ranked at position `i`
          (obtained by sorting `y_true` according to `y_pred`).
        - `gain_fn` is the user-provided function mapping relevance `y'_i` to a
          gain value. The default function (`default_gain_fn`) is typically
          equivalent to `lambda y: 2**y - 1`.
        - `rank_discount_fn` is the user-provided function mapping rank `i`
          (1-based) to a discount value. The default function
          (`default_rank_discount_fn`) is typically equivalent to
          `lambda rank: log2(rank + 1)`.
        - The final result aggregates these per-list scores, often involving
          normalization by list-specific weights derived from sample weights
          and gains, to produce a weighted average.
"""
extra_args = """
    gain_fn: callable. Maps relevance scores (`y_true`) to gain values. The
        default implements `2**y - 1`.
    rank_discount_fn: function. Maps rank positions (1-based) to discount
        values. The default (`default_rank_discount_fn`) typically implements
        `log2(rank + 1)`.
"""

DCG.__doc__ = ranking_metric_subclass_doc_string.format(
    metric_name="Discounted Cumulative Gain",
    metric_abbreviation="DCG",
    core_concept_sentence=core_concept_sentence,
    relevance_type_description=relevance_type_description,
    score_range_interpretation=score_range_interpretation,
    formula=formula,
    extra_args=extra_args,
)
