from typing import Any, Optional

import keras
from keras import ops

from keras_rs.src import types
from keras_rs.src.utils.keras_utils import check_rank
from keras_rs.src.utils.keras_utils import check_shapes_compatible
from keras_rs.src.utils.loss_and_metric_utils import process_inputs


class RankingMetric(keras.metrics.Mean):
    def __init__(self, k: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.k = k

    def compute_metric(
        self,
        y_true: types.Tensor,
        y_pred: types.Tensor,
        mask: Optional[types.Tensor] = None,
        sample_weight: Optional[types.Tensor] = None,
    ) -> types.Tensor:
        raise NotImplementedError(
            "All subclasses of the `keras_rs.losses.metrics.RankingMetric`"
            "must implement the `compute_metric()` method."
        )

    def update_state(
        self,
        y_true: types.Tensor,
        y_pred: types.Tensor,
        sample_weight: Optional[types.Tensor] = None,
    ) -> None:
        # === Process `sample_weight` ===
        if sample_weight is None:
            sample_weight = ops.cast(1, dtype=y_pred.dtype)

        # Check rank for `sample_weight`. Should be between 0 and `y_true_rank`.
        y_true_shape = ops.shape(y_true)
        y_true_rank = len(y_true_shape)
        sample_weight_shape = ops.shape(sample_weight)
        sample_weight_rank = len(sample_weight_shape)

        # Check `y_true_rank` first.
        check_rank(y_true_rank, allowed_ranks=(1, 2), tensor_name="y_true")

        check_rank(
            sample_weight_rank,
            allowed_ranks=tuple(range(y_true_rank + 1)),
            tensor_name="sample_weight",
        )

        if sample_weight_rank == 1:
            check_shapes_compatible(sample_weight_shape, (y_true_shape[0],))
        elif sample_weight_rank == 2:
            check_shapes_compatible(
                sample_weight_shape,
                y_true_shape,
            )

        sample_weight = ops.multiply(ops.ones_like(y_true), sample_weight)

        # === Process inputs - shape checking, upranking, etc. ===
        mask = ops.greater(
            sample_weight, ops.cast(0, dtype=sample_weight.dtype)
        )
        y_true, y_pred, mask = process_inputs(
            y_true=y_true,
            y_pred=y_pred,
            mask=mask,
            check_y_true_rank=False,
        )

        # === Actual computation ===
        per_list_metric_values, per_list_metric_weights = self.compute_metric(
            y_true=y_true, y_pred=y_pred, mask=mask, sample_weight=sample_weight
        )

        super().update_state(
            per_list_metric_values, sample_weight=per_list_metric_weights
        )
