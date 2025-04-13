from typing import Any, Callable, Optional

from keras import ops

from keras_rs.src import types
from keras_rs.src.api_export import keras_rs_export
from keras_rs.src.metrics.ranking_metric import RankingMetric
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
