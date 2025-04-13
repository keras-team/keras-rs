from keras import ops

from keras_rs.src import types
from keras_rs.src.api_export import keras_rs_export
from keras_rs.src.metrics.ranking_metric import RankingMetric
from keras_rs.src.metrics.ranking_metric import (
    ranking_metric_subclass_doc_string,
)
from keras_rs.src.utils.ranking_metrics_utils import get_list_weights
from keras_rs.src.utils.ranking_metrics_utils import sort_by_scores


@keras_rs_export("keras_rs.metrics.MeanAveragePrecision")
class MeanAveragePrecision(RankingMetric):
    def compute_metric(
        self,
        y_true: types.Tensor,
        y_pred: types.Tensor,
        mask: types.Tensor,
        sample_weight: types.Tensor,
    ) -> types.Tensor:
        relevance = ops.cast(
            ops.greater_equal(y_true, ops.cast(1, dtype=y_true.dtype)),
            dtype="float32",
        )
        sorted_relevance, sorted_weights = sort_by_scores(
            tensors_to_sort=[relevance, sample_weight],
            scores=y_pred,
            mask=mask,
            k=self.k,
        )
        per_list_relevant_counts = ops.cumsum(sorted_relevance, axis=1)
        per_list_cutoffs = ops.cumsum(ops.ones_like(sorted_relevance), axis=1)
        per_list_precisions = ops.divide_no_nan(
            per_list_relevant_counts, per_list_cutoffs
        )

        total_precision = ops.sum(
            ops.multiply(
                per_list_precisions,
                ops.multiply(sorted_weights, sorted_relevance),
            ),
            axis=1,
            keepdims=True,
        )

        # Compute the total relevance.
        total_relevance = ops.sum(
            ops.multiply(sample_weight, relevance), axis=1, keepdims=True
        )

        per_list_map = ops.divide_no_nan(total_precision, total_relevance)

        per_list_weights = get_list_weights(sample_weight, relevance)

        return per_list_map, per_list_weights


core_concept_sentence = (
    "an average of precision values computed after each relevant item is "
    "encountered in the ranked list"
)
relevance_type_description = "binary indicators (0 or 1) of relevance"
score_range_interpretation = (
    "from 0 to 1, with higher values indicating that relevant items are "
    "generally positioned higher in the ranking"
)

formula = """
    ```
    MAP(y, s) = sum_k (P@k(y, s) * rel(k)) / sum_i y_i
    rel(k) = y_i if rank(s_i) = k
    ```

    where:
        - `k` represents the rank position (starting from 1).
        - `sum_k` indicates a summation over all ranks `k` from 1 up to the list
          size (or the specified cutoff).
        - `P@k(y, s)` denotes the Precision at rank `k`, calculated as the
          number of relevant items found within the top `k` positions divided by
          `k`.
        - `rel(k)` represents the relevance of the item specifically at rank
          `k`. `rel(k)` is 1 if the item at rank `k` is relevant, and 0
          otherwise.
        - `y_i` is the true relevance label of the original item `i` before
          ranking.
        - `rank(s_i)` is the rank position assigned to item `i` based on its
          score `s_i`.
        - `sum_i y_i` calculates the total number of relevant items in the
          original list `y`.
        - If `sum_i y_i` (the total number of relevant items) is 0, the score
          for this query (Average Precision) is defined as 0.
"""
extra_args = ""

MeanAveragePrecision.__doc__ = ranking_metric_subclass_doc_string.format(
    metric_name="Mean Average Precision",
    metric_abbreviation="MAP",
    core_concept_sentence=core_concept_sentence,
    relevance_type_description=relevance_type_description,
    score_range_interpretation=score_range_interpretation,
    formula=formula,
    extra_args=extra_args,
)
