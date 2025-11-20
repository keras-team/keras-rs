import math
import os

import keras
import tensorflow as tf
from absl.testing import absltest
from absl.testing import parameterized
from keras import ops
from keras.layers import deserialize
from keras.layers import serialize

from keras_rs.src import testing
from keras_rs.src.layers.embedding.embed_reduce import EmbedReduce
from keras_rs.src.utils import tpu_test_utils

try:
    import jax
    from jax.experimental import sparse as jax_sparse
except ImportError:
    jax = None
    jax_sparse = None


class EmbedReduceTest(testing.TestCase, parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.on_tpu = "TPU_NAME" in os.environ

        if keras.backend.backend() == "tensorflow":
            tf.debugging.disable_traceback_filtering()

        self._strategy = tpu_test_utils.get_tpu_strategy(self)

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

        if self.on_tpu and input_type in ["ragged", "sparse"]:
            self.skipTest("Ragged and sparse are not compilable on TPU.")

        batch_size = 2 * self._strategy.num_replicas_in_sync

        def repeat_input(item, times):
            return [item[i % len(item)] for i in range(times)]

        if input_type == "dense" and input_rank == 1:
            inputs = ops.convert_to_tensor(repeat_input([1, 2], batch_size))
            weights = ops.convert_to_tensor(
                repeat_input([1.0, 2.0], batch_size)
            )
        elif input_type == "dense" and input_rank == 2:
            inputs = ops.convert_to_tensor(
                repeat_input([[1, 2], [3, 4]], batch_size)
            )
            weights = ops.convert_to_tensor(
                repeat_input([[1.0, 2.0], [3.0, 4.0]], batch_size)
            )
        elif input_type == "ragged" and input_rank == 2:
            inputs = tf.ragged.constant(
                repeat_input([[1], [2, 3, 4, 5]], batch_size)
            )
            weights = tf.ragged.constant(
                repeat_input([[1.0], [1.0, 2.0, 3.0, 4.0]], batch_size)
            )
        elif input_type == "sparse" and input_rank == 2:
            base_indices = [[0, 0], [1, 0], [1, 1], [1, 2], [1, 3]]
            base_values = [1, 2, 3, 4, 5]
            base_weights = [1.0, 1.0, 2.0, 3.0, 4.0]
            indices = []
            values = []
            weight_values = []
            for i in range(batch_size // 2):
                for idx, val, wgt in zip(
                    base_indices, base_values, base_weights
                ):
                    indices.append([i * 2 + idx[0], idx[1]])
                    values.append(val)
                    weight_values.append(wgt)

            if keras.backend.backend() == "tensorflow":
                inputs = tf.sparse.reorder(
                    tf.SparseTensor(indices, values, (batch_size, 4))
                )
                weights = tf.sparse.reorder(
                    tf.SparseTensor(indices, weight_values, (batch_size, 4))
                )
            elif keras.backend.backend() == "jax":
                inputs = jax_sparse.BCOO(
                    (ops.array(values), ops.array(indices)),
                    shape=(batch_size, 4),
                    unique_indices=True,
                )
                weights = jax_sparse.BCOO(
                    (ops.array(weight_values), ops.array(indices)),
                    shape=(batch_size, 4),
                    unique_indices=True,
                )

        if not use_weights:
            weights = None

        with self._strategy.scope():
            layer = EmbedReduce(10, 20, combiner=combiner)

        if keras.backend.backend() == "tensorflow":
            # TF requires weights to be None or match input type
            if input_type == "sparse" and not use_weights:
                res = tpu_test_utils.run_with_strategy(
                    self._strategy, layer.__call__, inputs
                )
            else:
                res = tpu_test_utils.run_with_strategy(
                    self._strategy, layer.__call__, inputs, weights
                )
        else:  # JAX or other
            res = layer(inputs, weights)

        self.assertEqual(res.shape, (batch_size, 20))

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

        expected = repeat_input(expected, batch_size)
        self.assertAllClose(res, expected)

    @parameterized.named_parameters(
        [
            (
                (
                    f"{input_type}_{input_rank}d"
                    f"{'_weights' if use_weights else ''}"
                ),
                input_type,
                input_rank,
                use_weights,
            )
            for input_type, input_rank in (
                ("dense", 1),
                ("dense", 2),
                ("ragged", 2),
                ("sparse", 2),
            )
            for use_weights in (False, True)
        ]
    )
    def test_symbolic_call(self, input_type, input_rank, use_weights):
        if input_type == "ragged" and keras.backend.backend() != "tensorflow":
            self.skipTest(f"ragged not supported on {keras.backend.backend()}")
        if input_type == "sparse":
            if keras.backend.backend() == "jax":
                self.assertTrue(
                    jax is not None, "JAX not found for JAX backend test."
                )
            elif keras.backend.backend() != "tensorflow":
                self.skipTest(
                    f"sparse not supported on {keras.backend.backend()}"
                )

        with self._strategy.scope():
            layer = EmbedReduce(10, 20, dtype="float32")

            input_tensor = keras.layers.Input(
                shape=(2,) if input_rank == 2 else (),
                sparse=input_type == "sparse",
                ragged=input_type == "ragged",
                dtype="int32",
            )

            if use_weights:
                weights = keras.layers.Input(
                    shape=(2,) if input_rank == 2 else (),
                    sparse=input_type == "sparse",
                    ragged=input_type == "ragged",
                    dtype="float32",
                )
                output = layer(input_tensor, weights)
            else:
                output = layer(input_tensor)

            self.assertEqual(output.shape, (None, 20))
            self.assertEqual(output.dtype, "float32")
            self.assertFalse(output.sparse)
            self.assertFalse(output.ragged)

    def test_predict(self):
        input_data = keras.random.randint((5, 7), minval=0, maxval=10)
        with self._strategy.scope():
            model = keras.models.Sequential([EmbedReduce(10, 20)])
            # Compilation is often needed for strategies to be fully utilized
            model.compile(optimizer="adam", loss="mse")

        # model.predict itself handles the strategy distribution
        model.predict(input_data, batch_size=2)

    def test_serialization(self):
        with self._strategy.scope():
            layer = EmbedReduce(10, 20, combiner="sqrtn")

        restored = deserialize(serialize(layer))
        self.assertDictEqual(layer.get_config(), restored.get_config())

    def test_model_saving(self):
        input_data = keras.random.randint((5, 7), minval=0, maxval=10)

        with self._strategy.scope():
            model = keras.models.Sequential([EmbedReduce(10, 20)])

        self.run_model_saving_test(
            model=model,
            input_data=input_data,
        )


if __name__ == "__main__":
    absltest.main()
