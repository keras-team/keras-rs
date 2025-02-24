import abc
from typing import Optional

import keras
from keras import ops

from keras_rs.src import types
from keras_rs.src.utils.loss_utils import pairwise_comparison


class PairwiseLoss(keras.losses.Loss):
    # TODO: Add `temperature`, `lambda_weights`.

    @abc.abstractmethod
    def pairwise_loss(self, pairwise_logits: types.Tensor) -> types.Tensor:
        raise NotImplementedError(
            "All subclasses of `keras_rs.losses.pairwise_loss.PairwiseLoss`"
            "must implement the `pairwise_loss()` method."
        )

    def compute_unreduced_loss(
        self,
        labels: types.Tensor,
        logits: types.Tensor,
        mask: Optional[types.Tensor] = None,
    ) -> tuple[types.Tensor, types.Tensor]:
        # If `mask` is not passed, mask all values less than 0 (since less than
        # 0 implies invalid labels).
        valid_mask = ops.greater_equal(labels, ops.cast(0.0, labels.dtype))

        if mask is not None:
            mask = ops.logical_and(valid_mask, mask)

        pairwise_labels, pairwise_logits = pairwise_comparison(
            labels=labels, logits=logits, mask=mask, logits_op=ops.subtract
        )

        return self.pairwise_loss(pairwise_logits), pairwise_labels

    def call(
        self,
        y_true: types.Tensor,
        y_pred: types.Tensor,
        sample_weight: Optional[types.Tensor] = None,
    ) -> types.Tensor:
        mask = None
        if isinstance(y_true, dict):
            if "labels" not in y_true:
                raise ValueError(
                    '`"labels"` should be present in `y_true`. Received: '
                    f"`y_true` = {y_true}"
                )

            mask = y_true.get("mask", None)
            y_true = y_true["labels"]

        losses, weights = self.compute_unreduced_loss(
            labels=y_true, logits=y_pred, mask=mask
        )
        losses = ops.multiply(losses, weights)
        losses = ops.sum(losses, axis=-1)
        return losses
