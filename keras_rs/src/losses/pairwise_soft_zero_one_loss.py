from keras import ops

from keras_rs.src import types
from keras_rs.src.api_export import keras_rs_export
from keras_rs.src.losses.pairwise_loss import PairwiseLoss


@keras_rs_export("keras_rs.losses.PairwiseSoftZeroOneLoss")
class PairwiseSoftZeroOneLoss(PairwiseLoss):
    def pairwise_loss(self, pairwise_logits: types.Tensor) -> types.Tensor:
        return ops.where(
            ops.greater(pairwise_logits, ops.array(0.0)),
            ops.subtract(ops.array(1.0), ops.sigmoid(pairwise_logits)),
            ops.sigmoid(ops.negative(pairwise_logits)),
        )
