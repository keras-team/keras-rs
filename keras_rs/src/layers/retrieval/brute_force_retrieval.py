from typing import Any

import keras

from keras_rs.src import types
from keras_rs.src.api_export import keras_rs_export
from keras_rs.src.layers.retrieval.retrieval import Retrieval


@keras_rs_export("keras_rs.layers.BruteForceRetrieval")
class BruteForceRetrieval(Retrieval):
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
        candidate_embeddings: types.Tensor | None = None,
        candidate_ids: types.Tensor | None = None,
        k: int = 10,
        return_scores: bool = True,
        **kwargs: Any,
    ) -> None:
        # Keep `k`, `return_scores` as separately passed args instead of keeping
        # them in `kwargs`. This is to ensure the user does not have to hop
        # to the base class to check which other args can be passed.
        super().__init__(k=k, return_scores=return_scores, **kwargs)

        self.candidate_embeddings = None
        self.candidate_ids = None

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
        candidate_ids: types.Tensor | None = None,
    ) -> None:
        """Update the set of candidates and optionally their candidate IDs.

        Args:
            candidate_embeddings: The candidate embeddings.
            candidate_ids: The identifiers for the candidates. If `None`, the
                indices of the candidates are returned instead.
        """
        self._validate_candidate_embeddings_and_ids(
            candidate_embeddings, candidate_ids
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
    ) -> types.Tensor | tuple[types.Tensor, types.Tensor]:
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
