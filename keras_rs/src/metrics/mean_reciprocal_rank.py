from keras import ops

from keras_rs.src import types
from keras_rs.src.api_export import keras_rs_export
from keras_rs.src.metrics.ranking_metric import RankingMetric
from keras_rs.src.metrics.ranking_metric import (
    ranking_metric_subclass_doc_string,
)
from keras_rs.src.metrics.ranking_metrics_utils import get_list_weights
from keras_rs.src.metrics.ranking_metrics_utils import sort_by_scores


@keras_rs_export("keras_rs.metrics.MeanReciprocalRank")
class MeanReciprocalRank(RankingMetric):
    def compute_metric(
        self,
        y_true: types.Tensor,
        y_pred: types.Tensor,
        mask: types.Tensor,
        sample_weight: types.Tensor,
    ) -> types.Tensor:
        # Assume: `y_true = [0, 0, 1]`, `y_pred = [0.1, 0.9, 0.2]`.
        # `sorted_y_true = [0, 1, 0]` (sorted in descending order).
        (sorted_y_true,) = sort_by_scores(
            tensors_to_sort=[y_true],
            scores=y_pred,
            mask=mask,
            k=self.k,
        )

        # This will depend on `k`, i.e., it will not always be the same as
        # `len(y_true)`.
        list_length = ops.shape(sorted_y_true)[1]

        # We consider only binary relevance here, anything above 1 is treated
        # as 1. `relevance = [0., 1., 0.]`.
        relevance = ops.cast(
            ops.greater_equal(
                sorted_y_true, ops.cast(1, dtype=sorted_y_true.dtype)
            ),
            dtype="float32",
        )

        # `reciprocal_rank = [1, 0.5, 0.33]`
        reciprocal_rank = ops.divide(
            ops.cast(1, dtype="float32"),
            ops.arange(1, list_length + 1, dtype="float32"),
        )

        # `mrr` should be of shape `(batch_size, 1)`.
        # `mrr = amax([0., 0.5, 0.]) = 0.5`
        mrr = ops.amax(
            ops.multiply(relevance, reciprocal_rank),
            axis=1,
            keepdims=True,
        )

        # Get weights.
        overall_relevance = ops.cast(
            ops.greater_equal(y_true, ops.cast(1, dtype=y_true.dtype)),
            dtype="float32",
        )
        per_list_weights = get_list_weights(
            weights=sample_weight, relevance=overall_relevance
        )

        return mrr, per_list_weights


core_concept_sentence = (
    "the rank position of the single highest-scoring relevant item"
)
relevance_type_description = "binary indicators (0 or 1) of relevance"
score_range_interpretation = (
    "from 0 to 1, with 1 indicating the first relevant item was always ranked "
    "first"
)
formula = """
    ```
    MRR(y, s) = max_{i} y_{i} / rank(s_{i})
    ```
"""
extra_args = ""
MeanReciprocalRank.__doc__ = ranking_metric_subclass_doc_string.format(
    metric_name="Mean Reciprocal Rank",
    metric_abbreviation="MRR",
    core_concept_sentence=core_concept_sentence,
    relevance_type_description=relevance_type_description,
    score_range_interpretation=score_range_interpretation,
    formula=formula,
    extra_args=extra_args,
)
