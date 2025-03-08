from keras import ops

from keras_rs.src import types
from keras_rs.src.api_export import keras_rs_export
from keras_rs.src.losses.pairwise_loss import PairwiseLoss
from keras_rs.src.losses.pairwise_loss import pairwise_loss_subclass_doc_string


@keras_rs_export("keras_rs.losses.PairwiseLogisticLoss")
class PairwiseLogisticLoss(PairwiseLoss):
    def pairwise_loss(self, pairwise_logits: types.Tensor) -> types.Tensor:
        return ops.add(
            ops.relu(ops.negative(pairwise_logits)),
            ops.log(
                ops.add(
                    ops.array(1),
                    ops.exp(ops.negative(ops.abs(pairwise_logits))),
                )
            ),
        )


formula = "loss = sum_{i} sum_{j} I(y_i > y_j) * log(1 + exp(-(s_i - s_j)))"
explanation = """
      - `log(1 + exp(-(s_i - s_j)))` is the logistic loss, which penalizes
        cases where the score difference `s_i - s_j` is not sufficiently large
        when `y_i > y_j`. This function provides a smooth approximation of the
        ideal step function, making it suitable for gradient-based optimization.
"""
extra_args = ""
PairwiseLogisticLoss.__doc__ = pairwise_loss_subclass_doc_string.format(
    formula=formula,
    explanation=explanation,
    extra_args=extra_args,
)
