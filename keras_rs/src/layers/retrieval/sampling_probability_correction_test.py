import keras
from absl.testing import parameterized
from keras import ops
from keras.layers import deserialize
from keras.layers import serialize

from keras_rs.src import testing
from keras_rs.src.layers.retrieval import sampling_probability_correction


class SamplingProbabilityCorrectionTest(
    testing.TestCase, parameterized.TestCase
):
    def setUp(self):
        super().setUp()

    def create_inputs(self, logits_rank=2, probs_rank=1):
        shape_3d = (15, 20, 10)
        logits_shape = shape_3d[-logits_rank:]
        probs_shape = shape_3d[-probs_rank:]

        rng = keras.random.SeedGenerator(42)
        logits = keras.random.uniform(logits_shape, seed=rng)
        probs = keras.random.uniform(probs_shape, seed=rng)
        return logits, probs

    @parameterized.named_parameters(
        [
            {
                "testcase_name": "logits_rank_1_probs_rank_1",
                "logits_rank": 1,
                "probs_rank": 1,
            },
            {
                "testcase_name": "logits_rank_2_probs_rank_1",
                "logits_rank": 2,
                "probs_rank": 1,
            },
            {
                "testcase_name": "logits_rank_2_probs_rank_2",
                "logits_rank": 2,
                "probs_rank": 2,
            },
            {
                "testcase_name": "logits_rank_3_probs_rank_1",
                "logits_rank": 3,
                "probs_rank": 1,
            },
            {
                "testcase_name": "logits_rank_3_probs_rank_2",
                "logits_rank": 3,
                "probs_rank": 2,
            },
            {
                "testcase_name": "logits_rank_3_probs_rank_3",
                "logits_rank": 3,
                "probs_rank": 3,
            },
        ]
    )
    def test_call(self, logits_rank, probs_rank):
        logits, probs = self.create_inputs(
            logits_rank=logits_rank, probs_rank=probs_rank
        )

        # Verifies logits are always less than corrected logits.
        layer = sampling_probability_correction.SamplingProbabilityCorrection()
        corrected_logits = layer(logits, probs)
        self.assertAllClose(
            ops.less(logits, corrected_logits), ops.ones(logits.shape)
        )

        # Set some of the probabilities to 0.
        probs_with_zeros = ops.multiply(
            probs,
            ops.cast(
                ops.greater_equal(keras.random.uniform(probs.shape), 0.5),
                dtype="float32",
            ),
        )

        # Verifies logits are always less than corrected logits.
        corrected_logits_with_zeros = layer(logits, probs_with_zeros)
        self.assertAllClose(
            ops.less(logits, corrected_logits_with_zeros),
            ops.ones(logits.shape),
        )

    def test_predict(self):
        # Note: for predict, we test with probabilities that have a batch dim.
        logits, probs = self.create_inputs(probs_rank=2)

        with self.strategy.scope():
            layer = (
                sampling_probability_correction.SamplingProbabilityCorrection()
            )
            in_logits = keras.layers.Input(logits.shape[1:])
            in_probs = keras.layers.Input(probs.shape[1:])
            out_logits = layer(in_logits, in_probs)
            model = keras.Model([in_logits, in_probs], out_logits)
            model.compile(optimizer="adam", loss="mse")

        model.predict([logits, probs], batch_size=4)

    def test_serialization(self):
        layer = sampling_probability_correction.SamplingProbabilityCorrection()
        restored = deserialize(serialize(layer))
        self.assertDictEqual(layer.get_config(), restored.get_config())

    def test_model_saving(self):
        logits, probs = self.create_inputs()

        layer = sampling_probability_correction.SamplingProbabilityCorrection()
        in_logits = keras.layers.Input(shape=logits.shape[1:])
        in_probs = keras.layers.Input(batch_shape=probs.shape)
        out_logits = layer(in_logits, in_probs)
        model = keras.Model([in_logits, in_probs], out_logits)

        self.run_model_saving_test(model=model, input_data=[logits, probs])
