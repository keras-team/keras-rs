import keras
import ml_dtypes
from keras import ops

from keras_rs.src import types
from keras_rs.src.api_export import keras_rs_export
from keras_rs.src.utils import keras_utils

SMALLEST_FLOAT = ml_dtypes.finfo("float32").smallest_normal / 100.0


@keras_rs_export("keras_rs.layers.RemoveAccidentalHits")
class RemoveAccidentalHits(keras.layers.Layer):
    """Zeroes the logits of accidental negatives.

    Zeroes the logits of negative candidates that have the same ID as the
    positive candidate in that row.

    Example:

    ```python
    # Create layer with the configured number of hard negatives to mine.
    remove_accidental_hits = keras_rs.layers.RemoveAccidentalHits()

    # This will zero the logits of negative candidates that have the same ID as
    # the positive candidate from `labels` so as to not negatively impact the
    # true positive.
    logits = remove_accidental_hits(logits, labels, candidate_ids)
    ```
    """

    def call(
        self,
        logits: types.Tensor,
        labels: types.Tensor,
        candidate_ids: types.Tensor,
    ) -> types.Tensor:
        """Zeroes selected logits.

        For each row in the batch, zeroes the logits of negative candidates that
        have the same ID as the positive candidate in that row.

        Args:
            logits: The logits tensor, typically `[batch_size, num_candidates]`
                but can have more dimensions or be 1D as `[num_candidates]`.
            labels: The one-hot labels tensor, must be the same shape as
                `logits`.
            candidate_ids: The candidate identifiers tensor, can be
                `[num_candidates]` or `[batch_size, num_candidates]` or have
                more dimensions as long as they match the last dimensions of
                `labels`.

        Returns:
            The modified logits with the same shape as the input logits.
        """
        # A more principled way is to implement
        # `softmax_cross_entropy_with_logits` with a input mask. Here we
        # approximate so by letting accidental hits have extremely small logits
        # (SMALLEST_FLOAT) for ease-of-implementation.

        labels_shape = ops.shape(labels)
        labels_rank = len(labels_shape)
        logits_shape = ops.shape(logits)
        candidate_ids_shape = ops.shape(candidate_ids)
        candidate_ids_rank = len(candidate_ids_shape)

        if not keras_utils.check_shapes_compatible(labels_shape, logits_shape):
            raise ValueError(
                "`labels` and `logits` should have the same shape. Received: "
                f"`labels.shape` = {labels_shape}, "
                f"`logits.shape` = {logits_shape}."
            )

        if not keras_utils.check_shapes_compatible(
            labels_shape[-candidate_ids_rank:], candidate_ids_shape
        ):
            raise ValueError(
                "`candidate_ids` should have the same shape as the last "
                "dimensions of `labels`. Received: "
                f"`candidate_ids.shape` = {candidate_ids_shape}, "
                f"`labels.shape` = {labels_shape}."
            )

        # Add dimensions to `candidate_ids` to have the same rank as `labels`.
        if candidate_ids_rank < labels_rank:
            candidate_ids = ops.expand_dims(
                candidate_ids, list(range(labels_rank - candidate_ids_rank))
            )
        positive_indices = ops.expand_dims(ops.argmax(labels, axis=-1), -1)
        positive_candidate_ids = ops.take(candidate_ids, positive_indices)

        duplicate = ops.cast(
            ops.equal(positive_candidate_ids, candidate_ids), labels.dtype
        )
        duplicate = ops.subtract(duplicate, labels)

        return ops.add(logits, ops.multiply(duplicate, SMALLEST_FLOAT))
