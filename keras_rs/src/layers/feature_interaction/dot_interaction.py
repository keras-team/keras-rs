from typing import Any

import keras
from keras import ops

from keras_rs.src import types
from keras_rs.src.api_export import keras_rs_export
from keras_rs.src.utils.keras_utils import check_shapes_compatible


@keras_rs_export("keras_rs.layers.DotInteraction")
class DotInteraction(keras.layers.Layer):
    """Dot interaction layer present in DLRM.

    This layer computes distinct dot product ("feature interactions") for every
    pair of features. If `self_interaction` is True, we calculate dot products
    of the form `dot(e_i, e_j)` for `i <= j`, and `dot(e_i, e_j)` for `i < j`,
    otherwise. `e_i` denotes representation of feature `i`. The layer can be
    used to build the DLRM model.

    Args:
        self_interaction: bool. Indicates whether features should
            "self-interact". If it is True, then the diagonal entries of the
            interaction matrix are also taken.
        skip_gather: bool. If it's set, then the upper triangular part of the
            interaction matrix is set to 0. The output will be of shape
            `[batch_size, num_features * num_features]` of which half of the
            entries will be zeros. Otherwise, the output will be only the lower
            triangular part of the interaction matrix. The latter saves space
            but is much slower.
        **kwargs: Args to pass to the base class.

    References:
    - [M. Naumov et al.](https://arxiv.org/abs/1906.00091)
    """

    def __init__(
        self,
        self_interaction: bool = False,
        skip_gather: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # Attributes.
        self.self_interaction = self_interaction
        self.skip_gather = skip_gather

    def _generate_tril_mask(
        self, pairwise_interaction_matrix: types.Tensor
    ) -> types.Tensor:
        """Generates lower triangular mask."""

        # If `self.self_interaction` is `True`, keep the main diagonal.
        k = -1
        if self.self_interaction:
            k = 0

        # Typecast k from Python int to tensor, because `ops.tril` uses
        # `tf.cond` (which requires tensors).
        # TODO (abheesht): Remove typecast once fix is merged in core Keras.
        if keras.config.backend() == "tensorflow":
            k = ops.array(k)
        tril_mask = ops.tril(
            ops.ones_like(pairwise_interaction_matrix, dtype=bool),
            k=k,
        )

        return tril_mask

    def _get_lower_triangular_indices(self, num_features: int) -> list[int]:
        """Python function which generates indices to get the lower triangular
        matrix as if it were flattened.
        """
        flattened_indices = []
        for i in range(num_features):
            k = i
            # if `self.self_interaction` is `True`, keep the main diagonal.
            if self.self_interaction:
                k += 1
            for j in range(k):
                flattened_index = i * num_features + j
                flattened_indices.append(flattened_index)

        return flattened_indices

    def call(self, inputs: list[types.Tensor]) -> types.Tensor:
        """Forward pass of the dot interaction layer.

        Args:
            inputs: list. Every element in the list represents a feature tensor
                of shape `[batch_size, feature_dim]`. All tensors in the list
                must have the same shape.

        Returns:
            Tensor representing feature interactions. The shape of the tensor is
            `[batch_size, k]` where `k` is
            `num_features * num_features` if `skip_gather` is `True`. Otherwise,
            `k` is `num_features * (num_features + 1) / 2` if
            `self_interaction` is `True`, and
            `num_features * (num_features - 1) / 2` if not.
        """

        # Check if all feature tensors have the same shape and are of rank 2.
        shape = ops.shape(inputs[0])
        for idx, tensor in enumerate(inputs):
            other_shape = ops.shape(tensor)

            if len(shape) != 2:
                raise ValueError(
                    "All feature tensors inside `inputs` should have rank 2. "
                    f"Received rank {len(shape)} at index {idx}."
                )

            if not check_shapes_compatible(shape, other_shape):
                raise ValueError(
                    "All feature tensors in `inputs` should have the same "
                    f"shape. Found at least one conflict: shape = {shape} at "
                    f"index 0 and shape = {ops.shape(tensor)} at index {idx}."
                )

        # `(batch_size, num_features, feature_dim)`
        features = ops.stack(inputs, axis=1)

        batch_size, num_features, _ = ops.shape(features)

        # Compute the dot product to get feature interactions. The shape here is
        # `(batch_size, num_features, num_features)`.
        pairwise_interaction_matrix = ops.matmul(
            features, ops.transpose(features, axes=(0, 2, 1))
        )

        # Set the upper triangle entries to 0, if `self.skip_gather` is True.
        # Else, "pick" only the lower triangle entries.
        if self.skip_gather:
            tril_mask = self._generate_tril_mask(pairwise_interaction_matrix)

            activations = ops.multiply(
                pairwise_interaction_matrix,
                ops.cast(tril_mask, dtype=pairwise_interaction_matrix.dtype),
            )
            # Rank-2 tensor.
            activations = ops.reshape(
                activations, (batch_size, num_features * num_features)
            )
        else:
            flattened_indices = self._get_lower_triangular_indices(num_features)
            pairwise_interaction_matrix_flattened = ops.reshape(
                pairwise_interaction_matrix,
                (batch_size, num_features * num_features),
            )
            activations = ops.take(
                pairwise_interaction_matrix_flattened,
                flattened_indices,
                axis=-1,
            )

        return activations

    def compute_output_shape(
        self, input_shape: list[types.TensorShape]
    ) -> types.TensorShape:
        num_features = len(input_shape)
        batch_size = input_shape[0][0]

        # Determine the number of pairwise interactions
        if self.self_interaction:
            output_dim = num_features * (num_features + 1) // 2
        else:
            output_dim = num_features * (num_features - 1) // 2

        if self.skip_gather:
            output_dim = num_features * num_features

        return (batch_size, output_dim)

    def get_config(self) -> dict[str, Any]:
        config: dict[str, Any] = super().get_config()

        config.update(
            {
                "self_interaction": self.self_interaction,
                "skip_gather": self.skip_gather,
            }
        )

        return config
