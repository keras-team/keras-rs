import abc
from typing import Any

import keras

from keras_rs.src import types
from keras_rs.src.api_export import keras_rs_export


@keras_rs_export("keras_rs.layers.Retrieval")
class Retrieval(keras.layers.Layer, abc.ABC):
    """Retrieval base abstract class.

    This layer provides a common interface for all retrieval layers. In order
    to implement a custom retrieval layer, this abstract class should be
    subclassed.

    Args:
        k: int. Number of candidates to retrieve.
        return_scores: bool. When `True`, this layer returns a tuple with the
            top scores and the top identifiers. When `False`, this layer returns
            a single tensor with the top identifiers.
    """

    def __init__(
        self,
        k: int = 10,
        return_scores: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.k = k
        self.return_scores = return_scores

    def _validate_candidate_embeddings_and_ids(
        self,
        candidate_embeddings: types.Tensor,
        candidate_ids: types.Tensor | None = None,
    ) -> None:
        """Validates inputs to `update_candidates()`."""

        if candidate_embeddings is None:
            raise ValueError("`candidate_embeddings` is required.")

        if len(candidate_embeddings.shape) != 2:
            raise ValueError(
                "`candidate_embeddings` must be a tensor of rank 2 "
                "(num_candidates, embedding_size), received "
                "`candidate_embeddings` with shape "
                f"{candidate_embeddings.shape}"
            )

        if candidate_embeddings.shape[0] < self.k:
            raise ValueError(
                "The number of candidates provided "
                f"({candidate_embeddings.shape[0]}) is less than the number of "
                f"candidates to retrieve (k={self.k})."
            )

        if (
            candidate_ids is not None
            and candidate_ids.shape[0] != candidate_embeddings.shape[0]
        ):
            raise ValueError(
                "The `candidate_embeddings` and `candidate_is` tensors must "
                "have the same number of rows, got tensors of shape "
                f"{candidate_embeddings.shape} and {candidate_ids.shape}."
            )

    @abc.abstractmethod
    def update_candidates(
        self,
        candidate_embeddings: types.Tensor,
        candidate_ids: types.Tensor | None = None,
    ) -> None:
        """Update the set of candidates and optionally their candidate IDs.

        Args:
            candidate_embeddings: The candidate embeddings.
            candidate_ids: The identifiers for the candidates. If `None`, the
                indices of the candidates are returned instead.
        """
        pass

    @abc.abstractmethod
    def call(
        self, inputs: types.Tensor
    ) -> types.Tensor | tuple[types.Tensor, types.Tensor]:
        """Returns the top candidates for the query passed as input.

        Args:
            inputs: the query for which to return top candidates.

        Returns:
            A tuple with the top scores and the top identifiers if
            `returns_scores` is True, otherwise a tensor with the top
            identifiers.
        """
        pass

    def compute_score(
        self, query_embedding: types.Tensor, candidate_embedding: types.Tensor
    ) -> types.Tensor:
        """Computes the standard dot product score from queries and candidates.

        Args:
            query_embedding: Tensor of query embedding corresponding to the
                queries for which to retrieve top candidates.
            candidate_embedding: Tensor of candidate embeddings.

        Returns:
            The dot product of queries and candidates.
        """

        return keras.ops.matmul(
            query_embedding, keras.ops.transpose(candidate_embedding)
        )

    def get_config(self) -> dict[str, Any]:
        config: dict[str, Any] = super().get_config()
        config.update(
            {
                "k": self.k,
                "return_scores": self.compute_score,
            }
        )
        return config
