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


@keras_rs_export("keras_rs.metrics.nDCG")
class nDCG(RankingMetric):
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

        weighted_gains = ops.multiply(
            sample_weight,
            self.gain_fn(y_true),
        )
        ideal_sorted_y_true, ideal_sorted_weights = sort_by_scores(
            tensors_to_sort=[y_true, sample_weight],
            scores=weighted_gains,
            k=self.k,
            mask=mask,
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


core_concept_sentence = (
    "A normalized measure of ranking quality based on Discounted Cumulative "
    "Gain (DCG), comparing the actual DCG to the Ideal DCG (IDCG) achievable "
    "for the given set of items at a specific cutoff `k`."
)
relevance_type_description = (
    "graded relevance scores (non-negative numbers where higher values "
    "indicate greater relevance)"
)
score_range_interpretation = (
    "Returns a weighted average score per ranked list, normalized ideally "
    "between 0 and 1. A score of 1 represents the perfect ranking according "
    "to true relevance (within the top-k), while 0 typically represents a "
    "ranking with no relevant items in the top-k. Higher scores indicate "
    "better ranking quality relative to the best possible ranking."
)

formula = """
    The metric calculates a weighted average nDCG score per list.
    For a single list, nDCG is computed as the ratio of the Discounted
    Cumulative Gain (DCG) of the predicted ranking to the Ideal Discounted
    Cumulative Gain (IDCG) of the best possible ranking:

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

    Definitions:
        - `k` is the rank position cutoff (determined by the `k` parameter or
          list size).
        - The sums are over the top `k` ranks `i` (from 1 to `k`).
        - `y'_i`: True relevance of the item at rank `i` in
          the ranking induced by `y_pred`.
        - `y''_i` True relevance of the item at rank `i` in
          the *ideal* ranking (sorted by `y_true` descending).
        - `gain_fn` is the user-provided function mapping relevance to gain.
          The default function (`default_gain_fn`) is typically equivalent to
          `lambda y: 2**y - 1`.
        - `rank_discount_fn` is the user-provided function mapping rank `i`
          (1-based) to a discount value. The default function
          (`default_rank_discount_fn`) is typically equivalent to
          `lambda rank: log2(rank + 1)`.
        - If IDCG@k is 0 (e.g., no relevant items), nDCG@k is defined as 0.
        - The final result often aggregates these per-list nDCG scores,
          potentially involving normalization by list-specific weights, to
          produce a weighted average.
"""
extra_args = """
    gain_fn: callable. Maps relevance scores (`y_true`) to gain values. The
        default implements `2**y - 1`. Used for both DCG and IDCG.
    rank_discount_fn: callable. Maps rank positions (1-based) to discount
        values. The default (`default_rank_discount_fn`) typically implements
        `log2(rank + 1)`. Used for both DCG and IDCG.
"""

nDCG.__doc__ = ranking_metric_subclass_doc_string.format(
    metric_name="Normalised Discounted Cumulative Gain",
    metric_abbreviation="nDCG",
    core_concept_sentence=core_concept_sentence,
    relevance_type_description=relevance_type_description,
    score_range_interpretation=score_range_interpretation,
    formula=formula,
    extra_args=extra_args,
)
