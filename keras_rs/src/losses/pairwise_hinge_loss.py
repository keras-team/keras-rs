from keras import ops

from keras_rs.src import types
from keras_rs.src.api_export import keras_rs_export
from keras_rs.src.losses.pairwise_loss import PairwiseLoss
from keras_rs.src.losses.pairwise_loss import pairwise_loss_subclass_doc_string


@keras_rs_export("keras_rs.losses.PairwiseHingeLoss")
class PairwiseHingeLoss(PairwiseLoss):
    def pairwise_loss(self, pairwise_logits: types.Tensor) -> types.Tensor:
        return ops.relu(ops.subtract(ops.array(1), pairwise_logits))


formula = "loss = sum_{i} sum_{j} I(y_i > y_j) * max(0, 1 - (s_i - s_j))"
explanation = """
      - `max(0, 1 - (s_i - s_j))` is the hinge loss, which penalizes cases where
        the score difference `s_i - s_j` is not sufficiently large when
        `y_i > y_j`.
"""
extra_args = ""
PairwiseHingeLoss.__doc__ = pairwise_loss_subclass_doc_string.format(
    formula=formula,
    explanation=explanation,
    extra_args=extra_args,
)
