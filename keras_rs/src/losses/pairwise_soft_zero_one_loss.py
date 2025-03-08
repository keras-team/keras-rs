from keras import ops

from keras_rs.src import types
from keras_rs.src.api_export import keras_rs_export
from keras_rs.src.losses.pairwise_loss import PairwiseLoss
from keras_rs.src.losses.pairwise_loss import pairwise_loss_subclass_doc_string


@keras_rs_export("keras_rs.losses.PairwiseSoftZeroOneLoss")
class PairwiseSoftZeroOneLoss(PairwiseLoss):
    def pairwise_loss(self, pairwise_logits: types.Tensor) -> types.Tensor:
        return ops.where(
            ops.greater(pairwise_logits, ops.array(0.0)),
            ops.subtract(ops.array(1.0), ops.sigmoid(pairwise_logits)),
            ops.sigmoid(ops.negative(pairwise_logits)),
        )


formula = "loss = sum_{i} sum_{j} I(y_i > y_j) * (1 - sigmoid(s_i - s_j))"
explanation = """
      - `(1 - sigmoid(s_i - s_j))` represents the soft zero-one loss, which
        approximates the ideal zero-one loss (which would be 1 if `s_i < s_j`
        and 0 otherwise) with a smooth, differentiable function. This makes it
        suitable for gradient-based optimization.
"""
extra_args = ""
PairwiseSoftZeroOneLoss.__doc__ = pairwise_loss_subclass_doc_string.format(
    formula=formula,
    explanation=explanation,
    extra_args=extra_args,
)
