from typing import Any, List, Tuple

import keras
from keras import ops

from keras_rs.src import types
from keras_rs.src.api_export import keras_rs_export


@keras_rs_export("keras_rs.layers.DotInteraction")
class DotInteraction(keras.layers.Layer):
    """Dot interaction layer present in DLRM.

    This layer computes distinct dot product ("feature interactions") for every
    pair of features. If `self_interaction` is True, we calculate dot products
    of the form `dot(e_i, e_j)` for `i <= j`, and `dot(e_i, e_j)` for `i < j`,
    otherwise. `e_i` represents features. The layer can be used to build
    the DLRM model.

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

        # Passed args.
        self.self_interaction = self_interaction
        self.skip_gather = skip_gather

    def call(self, inputs: List[types.Tensor]) -> types.Tensor:
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

        # A `ValueError` will be raised if the feature tensors inside the list
        # are not of equal shape. So, we do not need to handle the error here.
        features = ops.array(inputs)

        # Rearrange axes so as to have `batch_size` as the 0th dimension.
        features = ops.transpose(features, axes=(1, 0, 2))

        batch_size, _, _ = ops.shape(features)

        # Compute the dot product to get feature interactions. The shape here is
        # `(batch_size, num_features, num_features)`.
        pairwise_interaction_matrix = ops.matmul(
            features, ops.transpose(features, axes=(0, 2, 1))
        )

        # if `self.self_interaction` is `True`, keep the main diagonal.
        k = -1
        if self.self_interaction:
            k = 0

        tril_mask = ops.tril(
            ops.ones_like(pairwise_interaction_matrix, dtype=bool),
            k=k,
        )

        # Set the upper triangle entries to 0, if `self.skip_gather` is True.
        # Else, "pick" only the lower triangle entries.
        if self.skip_gather:
            activations = ops.multiply(
                pairwise_interaction_matrix,
                ops.cast(tril_mask, dtype=pairwise_interaction_matrix.dtype),
            )
        else:
            activations = pairwise_interaction_matrix[tril_mask]

        # Rank-2 tensor.
        activations = ops.reshape(activations, (batch_size, -1))

        return activations

    def compute_output_shape(
        self, input_shape: List[Tuple[int, int]]
    ) -> Tuple[int, int]:
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
