from keras import ops

from keras_rs.src import types
from keras_rs.src.metrics.ranking_metric import RankingMetric
from keras_rs.src.utils.ranking_metrics_utils import get_list_weights
from keras_rs.src.utils.ranking_metrics_utils import sort_by_scores


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
            y_pred=y_pred,
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
