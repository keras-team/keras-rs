import keras
import tensorflow as tf
from absl.testing import absltest
from absl.testing import parameterized

from keras_rs.src import testing
from keras_rs.src.layers.retrieval.retrieval import Retrieval
from keras_rs.src.utils import tpu_test_utils


class DummyRetrieval(Retrieval):
    def update_candidates(self, candidate_embeddings, candidate_ids=None):
        pass

    def call(self, inputs):
        pass


class RetrievalTest(testing.TestCase, parameterized.TestCase):
    def setUp(self):
        super().setUp()
        if keras.backend.backend() == "tensorflow":
            tf.debugging.disable_traceback_filtering()

        self._strategy = tpu_test_utils.get_tpu_strategy(self)
        with self._strategy.scope():
            self.layer = DummyRetrieval(k=5)

    @parameterized.named_parameters(
        ("embeddings_none", None, None, "`candidate_embeddings` is required."),
        (
            "embeddings_rank_1",
            keras.random.normal(shape=(10,)),
            None,
            "`candidate_embeddings` must be a tensor of rank 2",
        ),
        (
            "embeddings_smaller_than_k",
            keras.random.normal(shape=(3, 10)),
            None,
            "The number of candidates provided \\(3\\) is less than",
        ),
        (
            "embeddings_ids_shape",
            keras.random.normal(shape=(6, 10)),
            keras.random.randint(shape=(4,), minval=0, maxval=3),
            "The `candidate_embeddings` and `candidate_is` tensors must have "
            "the same number of rows",
        ),
    )
    def test_validate_candidate_embeddings_and_ids(
        self, candidate_embeddings, candidate_ids, error_msg
    ):
        with self.assertRaisesRegex(ValueError, error_msg):
            self.layer._validate_candidate_embeddings_and_ids(
                candidate_embeddings, candidate_ids
            )

    def test_call_not_overridden(self):
        with self._strategy.scope():

            class DummyRetrieval(Retrieval):
                def update_candidates(
                    self, candidate_embeddings, candidate_ids=None
                ):
                    pass

        with self.assertRaises(TypeError):
            DummyRetrieval(k=5)

    def test_update_candidates_not_overridden(self):
        with self._strategy.scope():

            class DummyRetrieval(Retrieval):
                def call(self, inputs):
                    pass

        with self.assertRaises(TypeError):
            DummyRetrieval(k=5)


if __name__ == "__main__":
    absltest.main()
