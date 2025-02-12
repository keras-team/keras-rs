from typing import Any

import keras
import numpy as np
from keras import ops

from keras_rs.src import types
from keras_rs.src.api_export import keras_rs_export

MAX_FLOAT = np.finfo(np.float32).max / 100.0


def _gather_elements_along_row(
    data: types.Tensor, column_indices: types.Tensor
) -> types.Tensor:
    """Gathers elements from a 2D tensor given the column indices of each row.

    First, gets the flat 1D indices to gather from. Then flattens the data to 1D
    and uses `ops.take()` to generate 1D output and finally reshapes the output
    back to 2D.

    Args:
        data: A [N, M] 2D `Tensor`.
        column_indices: A [N, K] 2D `Tensor` denoting for each row, the K column
            indices to gather elements from the data `Tensor`.

    Returns:
        A [N, K] `Tensor` including output elements gathered from data `Tensor`.

    Raises:
        ValueError: if the first dimensions of data and column_indices don't
            match.
    """
    num_row, num_column, *_ = ops.shape(data)
    num_gathered = ops.shape(column_indices)[1]
    row_indices = ops.tile(
        ops.expand_dims(ops.arange(num_row), -1), [1, num_gathered]
    )
    flat_data = ops.reshape(data, [-1])
    flat_indices = ops.reshape(
        ops.add(ops.multiply(row_indices, num_column), column_indices), [-1]
    )
    return ops.reshape(
        ops.take(flat_data, flat_indices), [num_row, num_gathered]
    )


@keras_rs_export("keras_rs.layers.HardNegativeMining")
class HardNegativeMining(keras.layers.Layer):
    """Transforms logits and labels to return hard negatives.

    Args:
        num_hard_negatives: How many hard negatives to return.
        **kwargs: Args to pass to the base class.
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
            logits: `[batch_size, number_of_candidates]` tensor of logits.
            labels: `[batch_size, number_of_candidates]` one-hot tensor of
                labels.

        Returns:
            tuple containing:
            - logits: `[batch_size, num_hard_negatives + 1]` tensor of logits.
            - labels: `[batch_size, num_hard_negatives + 1]` one-hot tensor of
                  labels.
        """

        # Number of sampled logits, i.e, the number of hard negatives to be
        # sampled (k) + number of true logit (1) per query, capped by batch
        # size.
        num_logits = ops.shape(logits)[1]
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
        _, col_indices = ops.top_k(
            ops.add(logits, ops.multiply(labels, MAX_FLOAT)),
            k=num_sampled,
            sorted=False,
        )

        # Gather sampled logits and corresponding labels.
        logits = _gather_elements_along_row(logits, col_indices)
        labels = _gather_elements_along_row(labels, col_indices)

        return logits, labels
