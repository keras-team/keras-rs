import math

import keras
from absl.testing import parameterized
from keras import ops
from keras.layers import deserialize
from keras.layers import serialize

from keras_rs.src import testing
from keras_rs.src.layers.embedding.embed_reduce import EmbedReduce


class EmbedReduceTest(testing.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        [
            (
                (
                    f"{combiner}_{input_type}_{input_rank}d"
                    f"{'_weights' if use_weights else ''}"
                ),
                combiner,
                input_type,
                input_rank,
                use_weights,
            )
            for combiner in ("sum", "mean", "sqrtn")
            for input_type, input_rank in (
                ("dense", 1),
                ("dense", 2),
                ("ragged", 2),
                ("sparse", 2),
            )
            for use_weights in (False, True)
        ]
    )
    def test_call(self, combiner, input_type, input_rank, use_weights):
        if input_type == "ragged" and keras.backend.backend() != "tensorflow":
            self.skipTest(f"ragged not supported on {keras.backend.backend()}")
        if input_type == "sparse" and keras.backend.backend() not in (
            "jax",
            "tensorflow",
        ):
            self.skipTest(f"sparse not supported on {keras.backend.backend()}")

        if input_type == "dense" and input_rank == 1:
            inputs = ops.convert_to_tensor([1, 2])
            weights = ops.convert_to_tensor([1.0, 2.0])
        elif input_type == "dense" and input_rank == 2:
            inputs = ops.convert_to_tensor([[1, 2], [3, 4]])
            weights = ops.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])
        elif input_type == "ragged" and input_rank == 2:
            import tensorflow as tf

            inputs = tf.ragged.constant([[1], [2, 3, 4, 5]])
            weights = tf.ragged.constant([[1.0], [1.0, 2.0, 3.0, 4.0]])
        elif input_type == "sparse" and input_rank == 2:
            indices = [[0, 0], [1, 0], [1, 1], [1, 2], [1, 3]]

            if keras.backend.backend() == "tensorflow":
                import tensorflow as tf

                inputs = tf.sparse.reorder(
                    tf.SparseTensor(indices, [1, 2, 3, 4, 5], (2, 4))
                )
                weights = tf.sparse.reorder(
                    tf.SparseTensor(indices, [1.0, 1.0, 2.0, 3.0, 4.0], (2, 4))
                )
            elif keras.backend.backend() == "jax":
                from jax.experimental import sparse as jax_sparse

                inputs = jax_sparse.BCOO(
                    ([1, 2, 3, 4, 5], indices),
                    shape=(2, 4),
                    unique_indices=True,
                )
                weights = jax_sparse.BCOO(
                    ([1.0, 1.0, 2.0, 3.0, 4.0], indices),
                    shape=(2, 4),
                    unique_indices=True,
                )

        if not use_weights:
            weights = None

        layer = EmbedReduce(10, 20, combiner=combiner)
        res = layer(inputs, weights)

        self.assertEqual(res.shape, (2, 20))

        e = layer.embeddings
        if input_type == "dense" and input_rank == 1:
            if combiner == "sum" and use_weights:
                expected = [e[1], e[2] * 2.0]
            else:
                expected = [e[1], e[2]]
        elif input_type == "dense" and input_rank == 2:
            if use_weights:
                expected = [e[1] + e[2] * 2.0, e[3] * 3.0 + e[4] * 4.0]
            else:
                expected = [e[1] + e[2], e[3] + e[4]]

            if combiner == "mean":
                expected[0] /= 3.0 if use_weights else 2.0
                expected[1] /= 7.0 if use_weights else 2.0
            elif combiner == "sqrtn":
                expected[0] /= math.sqrt(5.0 if use_weights else 2.0)
                expected[1] /= math.sqrt(25.0 if use_weights else 2.0)
        else:  # ragged, sparse and input_rank == 2
            if use_weights:
                expected = [e[1], e[2] + e[3] * 2.0 + e[4] * 3.0 + e[5] * 4.0]
            else:
                expected = [e[1], e[2] + e[3] + e[4] + e[5]]

            if combiner == "mean":
                expected[1] /= 10.0 if use_weights else 4.0
            elif combiner == "sqrtn":
                expected[1] /= math.sqrt(30.0 if use_weights else 4.0)

        self.assertAllClose(res, expected)

    def test_predict(self):
        input = keras.random.randint((5, 7), minval=0, maxval=10)
        model = keras.models.Sequential([EmbedReduce(10, 20)])
        model.predict(input, batch_size=2)

    def test_serialization(self):
        layer = EmbedReduce(10, 20, combiner="sqrtn")
        restored = deserialize(serialize(layer))
        self.assertDictEqual(layer.get_config(), restored.get_config())

    def test_model_saving(self):
        input = keras.random.randint((5, 7), minval=0, maxval=10)
        model = keras.models.Sequential([EmbedReduce(10, 20)])

        self.run_model_saving_test(
            model=model,
            input_data=input,
        )
