from typing import Any

import keras
import ml_dtypes
from keras import ops

from keras_rs.src import types
from keras_rs.src.api_export import keras_rs_export

MAX_FLOAT = ml_dtypes.finfo("float32").max / 100.0


@keras_rs_export("keras_rs.layers.HardNegativeMining")
class HardNegativeMining(keras.layers.Layer):
    """Filter logits and labels to return hard negatives.

    The output will include logits and labels for the requested number of hard
    negatives as well as the positive candidate.

    Args:
        num_hard_negatives: How many hard negatives to return.
        **kwargs: Args to pass to the base class.

    Example:

    ```python
    # Create layer with the configured number of hard negatives to mine.
    hard_negative_mining = keras_rs.layers.HardNegativeMining(
        num_hard_negatives=10
    )

    # This will retrieve the top 10 negative candidates plus the positive
    # candidate from `labels` for each row.
    out_logits, out_labels = hard_negative_mining(in_logits, in_labels)
    ```
    """

    def __init__(self, num_hard_negatives: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._num_hard_negatives = num_hard_negatives
        self.built = True

    def call(
        self, logits: types.Tensor, labels: types.Tensor
    ) -> tuple[types.Tensor, types.Tensor]:
        """Filters logits and labels with per-query hard negative mining.

        The result will include logits and labels for `num_hard_negatives`
        negatives as well as the positive candidate.

        Args:
            logits: The logits tensor, typically `[batch_size, num_candidates]`
                but can have more dimensions or be 1D as `[num_candidates]`.
            labels: The one-hot labels tensor, must be the same shape as
                `logits`.

        Returns:
            A tuple containing two tensors with the last dimension of
            `num_candidates` replaced with `num_hard_negatives + 1`.

        * logits: `[..., num_hard_negatives + 1]` tensor of logits.
        * labels: `[..., num_hard_negatives + 1]` one-hot tensor of labels.
        """

        # Number of sampled logits, i.e, the number of hard negatives to be
        # sampled (k) + number of true logit (1) per query, capped by batch
        # size.
        num_logits = ops.shape(logits)[-1]
        if isinstance(num_logits, int):
            num_sampled = min(self._num_hard_negatives + 1, num_logits)
        else:
            num_sampled = ops.minimum(self._num_hard_negatives + 1, num_logits)
        # To gather indices of top k negative logits per row (query) in logits,
        # true logits need to be excluded. First replace the true logits
        # (corresponding to positive labels) with a large score value and then
        # select the top k + 1 logits from each row so that selected indices
        # include the indices of true logit + top k negative logits. This
        # approach is to avoid using inefficient masking when excluding true
        # logits.

        # For each query, get the indices of the logits which have the highest
        # k + 1 logit values, including the highest k negative logits and one
        # true logit.
        _, indices = ops.top_k(
            ops.add(logits, ops.multiply(labels, MAX_FLOAT)),
            k=num_sampled,
            sorted=False,
        )

        # Gather sampled logits and corresponding labels.
        logits = ops.take_along_axis(logits, indices, axis=-1)
        labels = ops.take_along_axis(labels, indices, axis=-1)

        return logits, labels
