import keras
from absl.testing import parameterized

from keras_rs.src import testing
from keras_rs.src.layers.retrieval import brute_force_retrieval


class BruteForceRetrievalTest(testing.TestCase, parameterized.TestCase):
    @parameterized.product(
        has_identifiers=(True, False),
        return_scores=(True, False),
    )
    def test_brute_force_retrieval(self, has_identifiers, return_scores):
        num_candidates = 100
        num_queries = 16
        k = 20

        rng = keras.random.SeedGenerator(42)
        candidates = keras.random.normal(
            (num_candidates, 4), dtype="float32", seed=rng
        )
        candidate_indices = (
            keras.ops.arange(3, num_candidates + 3, dtype="int32")
            if has_identifiers
            else None
        )

        layer = brute_force_retrieval.BruteForceRetrieval(
            k=k,
            candidate_embeddings=candidates,
            candidate_ids=candidate_indices,
            return_scores=return_scores,
        )

        query = keras.random.normal((num_queries, 4), dtype="float32", seed=rng)
        scores = keras.ops.matmul(query, keras.ops.transpose(candidates))
        expected_top_indices = keras.ops.take(
            keras.ops.argsort(-scores, axis=1),
            keras.ops.arange(k),
            axis=1,
        )
        expected_top_scores = keras.ops.take_along_axis(
            scores, expected_top_indices, 1
        )
        if has_identifiers:
            expected_top_indices = keras.ops.take(
                candidate_indices, expected_top_indices, axis=0
            )

        # Call twice to ensure the results are repeatable.
        for i in range(2):
            if i:
                # First time uses values from __init__, second time uses update.
                layer.update_candidates(candidates, candidate_indices)

            if return_scores:
                top_scores, top_indices = layer(query)
                self.assertEqual(top_scores.shape, expected_top_scores.shape)
                self.assertAllClose(top_scores, expected_top_scores, atol=1e-4)
            else:
                top_indices = layer(query)

            self.assertEqual(top_indices.shape, expected_top_indices.shape)
            self.assertAllClose(top_indices, expected_top_indices)
