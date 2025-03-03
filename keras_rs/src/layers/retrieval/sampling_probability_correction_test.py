import keras
from keras import ops
from keras.layers import deserialize
from keras.layers import serialize

from keras_rs.src import testing
from keras_rs.src.layers.retrieval import sampling_probability_correction


class SamplingProbabilityCorrectionTest(testing.TestCase):
    def setUp(self):
        shape = (10, 20)  # (num_queries, num_candidates)
        rng = keras.random.SeedGenerator(42)
        self.logits = keras.random.uniform(shape, seed=rng)
        self.probs_1d = keras.random.uniform(shape[1:], seed=rng)
        self.probs_2d = keras.random.uniform(shape, seed=rng)

    def test_call(self):
        # Verifies logits are always less than corrected logits.
        layer = sampling_probability_correction.SamplingProbabilityCorrection()
        corrected_logits = layer(self.logits, self.probs_1d)
        self.assertAllClose(
            ops.less(self.logits, corrected_logits), ops.ones(self.logits.shape)
        )

        # Set some of the probabilities to 0.
        probs_with_zeros = ops.multiply(
            self.probs_1d,
            ops.cast(
                ops.greater_equal(
                    keras.random.uniform(self.probs_1d.shape), 0.5
                ),
                dtype="float32",
            ),
        )

        # Verifies logits are always less than corrected logits.
        corrected_logits_with_zeros = layer(self.logits, probs_with_zeros)
        self.assertAllClose(
            ops.less(self.logits, corrected_logits_with_zeros),
            ops.ones(self.logits.shape),
        )

    def test_predict(self):
        # Note: for predict, we test with probabilities that have a batch dim.
        layer = sampling_probability_correction.SamplingProbabilityCorrection()
        in_logits = keras.layers.Input(self.logits.shape[1:])
        in_probs = keras.layers.Input(self.probs_2d.shape[1:])
        out_logits = layer(in_logits, in_probs)
        model = keras.Model([in_logits, in_probs], out_logits)

        model.predict([self.logits, self.probs_2d], batch_size=4)

    def test_serialization(self):
        layer = sampling_probability_correction.SamplingProbabilityCorrection()
        restored = deserialize(serialize(layer))
        self.assertDictEqual(layer.get_config(), restored.get_config())

    def test_model_saving(self):
        layer = sampling_probability_correction.SamplingProbabilityCorrection()
        in_logits = keras.layers.Input(shape=self.logits.shape[1:])
        in_probs = keras.layers.Input(batch_shape=self.probs_1d.shape)
        out_logits = layer(in_logits, in_probs)
        model = keras.Model([in_logits, in_probs], out_logits)

        self.run_model_saving_test(
            model=model, input_data=[self.logits, self.probs_1d]
        )
