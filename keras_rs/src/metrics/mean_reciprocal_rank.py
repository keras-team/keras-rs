from typing import Optional

from keras import ops

from keras_rs.src import types
from keras_rs.src.metrics.ranking_metric import RankingMetric
from keras_rs.src.utils.ranking_metrics_utils import get_list_weights
from keras_rs.src.utils.ranking_metrics_utils import sort_by_scores


class MeanReciprocalRank(RankingMetric):
    def compute_metric(
        self,
        y_true: types.Tensor,
        y_pred: types.Tensor,
        mask: Optional[types.Tensor] = None,
        sample_weight: Optional[types.Tensor] = None,
    ) -> types.Tensor:
        sorted_y_true = sort_by_scores(
            y_true=y_true,
            y_pred=y_pred,
            mask=mask,
            k=self.k,
        )

        list_length = ops.shape(sorted_y_true)[1]

        # We consider only binary relevance here.
        relevance = ops.cast(
            ops.greater_equal(
                sorted_y_true, ops.cast(1, dtype=sorted_y_true.dtype)
            ),
            dtype="float32",
        )
        reciprocal_rank = ops.divide(
            ops.cast(1, dtype="float32"),
            ops.range(1, list_length + 1, dtype="float32"),
        )

        # `mrr` should be of shape `(batch_size, 1)`.
        mrr = ops.maximum(ops.multiply(relevance, reciprocal_rank), axis=1)
        mrr = ops.expand_dims(mrr, axis=1)

        # Get weights.
        overall_relevance = ops.cast(
            ops.greater_equal(y_true, ops.cast(1, dtype=y_true.dtype)),
            dtype="float32",
        )
        per_list_weights = get_list_weights(
            weights=sample_weight, relevance=overall_relevance
        )

        return mrr, per_list_weights
