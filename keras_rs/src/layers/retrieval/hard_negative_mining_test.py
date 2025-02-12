import keras
from absl.testing import parameterized
from keras import ops
from keras.layers import deserialize
from keras.layers import serialize

from keras_rs.src import testing
from keras_rs.src.layers.retrieval import hard_negative_mining


class HardNegativeMiningTest(testing.TestCase, parameterized.TestCase):
    @parameterized.parameters(42, 123, 8391, 12390, 1230)
    def test_call(self, random_seed):
        num_hard_negatives = 3
        # (num_queries, num_candidates)
        shape = (2, 20)
        rng = keras.random.SeedGenerator(random_seed)

        logits = keras.random.uniform(shape, dtype="float32", seed=rng)
        labels = ops.transpose(
            keras.random.shuffle(
                ops.transpose(ops.eye(*shape, dtype="float32")), seed=rng
            )
        )

        out_logits, out_labels = hard_negative_mining.HardNegativeMining(
            num_hard_negatives
        )(logits, labels)

        self.assertEqual(out_logits.shape[-1], num_hard_negatives + 1)

        # Logits for positives are always returned.
        self.assertAllClose(
            ops.sum(out_logits * out_labels, axis=1),
            ops.sum(logits * labels, axis=1),
        )

        # Set the logits for labels to be highest to ignore effect of labels.
        logits = logits + labels * 1000.0

        out_logits, _ = hard_negative_mining.HardNegativeMining(
            num_hard_negatives
        )(logits, labels)

        # Highest K logits are always returned.
        self.assertAllClose(
            ops.sort(logits, axis=1)[:, -num_hard_negatives - 1 :],
            ops.sort(out_logits),
        )

    def test_predict(self):
        num_candidates = 20
        in_logits = keras.layers.Input(shape=(num_candidates,))
        in_labels = keras.layers.Input(shape=(num_candidates,))
        out_logits, out_labels = hard_negative_mining.HardNegativeMining(
            num_hard_negatives=3
        )(in_logits, in_labels)
        model = keras.Model([in_logits, in_labels], [out_logits, out_labels])

        shape = (25, num_candidates)
        rng = keras.random.SeedGenerator(42)
        logits = keras.random.uniform(shape, dtype="float32", seed=rng)
        labels = ops.transpose(
            keras.random.shuffle(
                ops.transpose(ops.eye(*shape, dtype="float32")), seed=rng
            )
        )

        model.predict([logits, labels], batch_size=10)

    def test_serialization(self):
        layer = hard_negative_mining.HardNegativeMining(num_hard_negatives=3)
        restored = deserialize(serialize(layer))
        self.assertDictEqual(layer.get_config(), restored.get_config())

    def test_model_saving(self):
        num_candidates = 20
        in_logits = keras.layers.Input(shape=(num_candidates,))
        in_labels = keras.layers.Input(shape=(num_candidates,))
        out_logits, out_labels = hard_negative_mining.HardNegativeMining(
            num_hard_negatives=3
        )(in_logits, in_labels)
        model = keras.Model([in_logits, in_labels], [out_logits, out_labels])

        shape = (2, num_candidates)
        rng = keras.random.SeedGenerator(42)
        logits = keras.random.uniform(shape, dtype="float32", seed=rng)
        labels = ops.transpose(
            keras.random.shuffle(
                ops.transpose(ops.eye(*shape, dtype="float32")), seed=rng
            )
        )

        self.run_model_saving_test(
            model=model,
            input_data=[logits, labels],
        )
