from typing import Any, Optional, Union

import keras

from keras_rs.src import types
from keras_rs.src.api_export import keras_rs_export


@keras_rs_export("keras_rs.layers.BruteForceRetrieval")
class BruteForceRetrieval(keras.layers.Layer):
    """Brute force top-k retrieval.

    This layer maintains a set of candidates and is able to exactly retrieve the
    top-k candidates for a given query. It does this by computing the scores for
    all of the candidates for the query and extracting the top ones. The
    returned top-k candidates are sorted by score.

    By default, this layer returns a tuple with the top scores and top
    identifiers, but it can be configured to return a single tensor with the top
    identifiers.

    The identifiers for the candidates can be specified as a tensor. If not
    provided, the IDs used are simply the candidate indices.

    Note that the serialization of this layer does not preserve the candidates
    and only saves the `k` and `return_scores` arguments. One has to call
    `update_candidates` after deserializing the layers.

    Args:
        candidate_embeddings: The candidate embeddings. If `None`,
            candidates must be provided using `update_candidates` before
            using this layer.
        candidate_ids: The identifiers for the candidates. If `None` the
            indices of the candidates are returned instead.
        k: Number of candidates to retrieve.
        return_scores: When `True`, this layer returns a tuple with the top
            scores and the top identifiers. When `False`, this layer returns
            a single tensor with the top identifiers.
        **kwargs: Args to pass to the base class.

    Example:

    ```python
    retrieval = keras_rs.layers.BruteForceRetrieval(k=100)

    # At some later point, we update the candidates.
    retrieval.update_candidates(candidate_embeddings, candidate_ids)

    # We can then retrieve the top candidates for any number of queries.
    # Scores are stored highest first. Scores correspond to ids in the same row.
    tops_scores, top_ids = retrieval(query_embeddings)
    ```
    """

    def __init__(
        self,
        candidate_embeddings: Optional[types.Tensor] = None,
        candidate_ids: Optional[types.Tensor] = None,
        k: int = 10,
        return_scores: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.candidate_embeddings = None
        self.candidate_ids = None
        self.k = k
        self.return_scores = return_scores

        if candidate_embeddings is None:
            if candidate_ids is not None:
                raise ValueError(
                    "You cannot provide `candidate_ids` without providing "
                    "`candidate_embeddings`"
                )
        else:
            self.update_candidates(candidate_embeddings, candidate_ids)

    def update_candidates(
        self,
        candidate_embeddings: types.Tensor,
        candidate_ids: Optional[types.Tensor] = None,
    ) -> None:
        """Update the set of candidates and optionally their candidate IDs.

        Args:
            candidate_embeddings: The candidate embeddings.
            candidate_ids: The identifiers for the candidates. If `None` the
                indices of the candidates are returned instead.
        """
        if candidate_embeddings is None:
            raise ValueError("`candidate_embeddings` is required")

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

        if self.candidate_embeddings is not None:
            # Update of existing variables.
            self.candidate_embeddings.assign(candidate_embeddings)

            if self.candidate_ids is None:
                if candidate_ids is not None:
                    raise ValueError(
                        "New `candidate_ids` cannot be provided as previous "
                        "candidates did not have candidate IDs"
                    )
            else:
                if candidate_ids is not None:
                    self.candidate_ids.assign(candidate_ids)
        else:
            # Initial creation of variables.
            self.candidate_embeddings = keras.Variable(
                initializer=candidate_embeddings,
                name="candidate_embeddings",
                trainable=False,
            )
            if candidate_ids is not None:
                self.candidate_ids = keras.Variable(
                    initializer=candidate_ids,
                    dtype="int32",
                    name="candidate_ids",
                    trainable=False,
                )
        self.built = True

    def call(
        self, inputs: types.Tensor
    ) -> Union[types.Tensor, tuple[types.Tensor, types.Tensor]]:
        """Returns the top candidates for the query passed as input.

        Args:
            inputs: the query for which to return top candidates.

        Returns:
            A tuple with the top scores and the top identifiers if
            `returns_scores` is True, otherwise a tensor with the top
            identifiers.
        """
        scores = self.compute_score(inputs, self.candidate_embeddings)
        top_scores, top_ids = keras.ops.top_k(scores, k=self.k)

        if self.candidate_ids is not None:
            top_ids = keras.ops.take(self.candidate_ids, top_ids, axis=0)

        if self.return_scores:
            return top_scores, top_ids
        else:
            return top_ids

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
