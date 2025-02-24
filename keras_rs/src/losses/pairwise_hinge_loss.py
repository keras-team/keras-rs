from keras import ops

from keras_rs.src import types
from keras_rs.src.api_export import keras_rs_export
from keras_rs.src.losses.pairwise_loss import PairwiseLoss


@keras_rs_export("keras_rs.losses.PairwiseHingeLoss")
class PairwiseHingeLoss(PairwiseLoss):
    def pairwise_loss(self, pairwise_logits: types.Tensor) -> types.Tensor:
        return ops.relu(ops.subtract(ops.array(1), pairwise_logits))
