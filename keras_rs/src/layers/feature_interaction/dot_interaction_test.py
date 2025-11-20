import os

import keras
import tensorflow as tf
from absl.testing import absltest
from absl.testing import parameterized
from keras import ops
from keras.layers import deserialize
from keras.layers import serialize

from keras_rs.src import testing
from keras_rs.src.layers.feature_interaction.dot_interaction import (
    DotInteraction,
)
from keras_rs.src.utils import tpu_test_utils


class DotInteractionTest(testing.TestCase, parameterized.TestCase):
    def setUp(self):
        super().setUp()
        if keras.backend.backend() == "tensorflow":
            tf.debugging.disable_traceback_filtering()

        self.on_tpu = "TPU_NAME" in os.environ
        self._strategy = tpu_test_utils.get_tpu_strategy(self)

        self.input = [
            ops.array([[0.1, -4.3, 0.2, 1.1, 0.3]]),
            ops.array([[2.0, 3.2, -1.0, 0.0, 1.0]]),
            ops.array([[0.0, 1.0, -3.0, -2.2, -0.2]]),
        ]

        feature1 = self.input[0]
        feature2 = self.input[1]
        feature3 = self.input[2]

        f11 = ops.dot(feature1[0], feature1[0])
        f12 = ops.dot(feature1[0], feature2[0])
        f13 = ops.dot(feature1[0], feature3[0])
        f22 = ops.dot(feature2[0], feature2[0])
        f23 = ops.dot(feature2[0], feature3[0])
        f33 = ops.dot(feature3[0], feature3[0])

        exp_output1 = ops.array([[f12, f13, f23]])
        exp_output2 = ops.array([[0, 0, 0, f12, 0, 0, f13, f23, 0]])
        exp_output3 = ops.array([[f11, f12, f22, f13, f23, f33]])
        exp_output4 = ops.array(
            [
                [
                    f11,
                    0,
                    0,
                    f12,
                    f22,
                    0,
                    f13,
                    f23,
                    f33,
                ]
            ]
        )

        self.exp_outputs = [exp_output1, exp_output2, exp_output3, exp_output4]

    @parameterized.named_parameters(
        (
            "self_interaction_false_skip_gather_false",
            False,
            False,
            0,
        ),
        (
            "self_interaction_false_skip_gather_true",
            False,
            True,
            1,
        ),
        (
            "self_interaction_true_skip_gather_false",
            True,
            False,
            2,
        ),
        (
            "self_interaction_true_skip_gather_true",
            True,
            True,
            3,
        ),
    )
    def test_call(self, self_interaction, skip_gather, exp_output_idx):
        with self._strategy.scope():
            layer = DotInteraction(
                self_interaction=self_interaction, skip_gather=skip_gather
            )
        output = tpu_test_utils.run_with_strategy(
            self._strategy, layer, self.input
        )
        self.assertAllClose(
            output, self.exp_outputs[exp_output_idx], is_tpu=self.on_tpu
        )

    def test_invalid_input_rank(self):
        rank_1_input = [ops.ones((3,)), ops.ones((3,))]

        with self._strategy.scope():
            layer = DotInteraction()
        with self.assertRaises(ValueError):
            tpu_test_utils.run_with_strategy(
                self._strategy, layer, rank_1_input
            )

    def test_invalid_input_different_shapes(self):
        unequal_shape_input = [ops.ones((1, 3)), ops.ones((1, 4))]

        with self._strategy.scope():
            layer = DotInteraction()
        with self.assertRaises(ValueError):
            tpu_test_utils.run_with_strategy(
                self._strategy, layer, unequal_shape_input
            )

    @parameterized.named_parameters(
        (
            "self_interaction_false_skip_gather_false",
            False,
            False,
        ),
        (
            "self_interaction_false_skip_gather_true",
            False,
            True,
        ),
        (
            "self_interaction_true_skip_gather_false",
            True,
            False,
        ),
        (
            "self_interaction_true_skip_gather_true",
            True,
            True,
        ),
    )
    def test_predict(self, self_interaction, skip_gather):
        with self._strategy.scope():
            feature1 = keras.layers.Input(shape=(5,))
            feature2 = keras.layers.Input(shape=(5,))
            feature3 = keras.layers.Input(shape=(5,))
            x = DotInteraction(
                self_interaction=self_interaction, skip_gather=skip_gather
            )([feature1, feature2, feature3])
            x = keras.layers.Dense(units=1)(x)
            model = keras.Model([feature1, feature2, feature3], x)
            model.compile(optimizer="adam", loss="mse")

        model.predict(self.input, batch_size=2)

    def test_serialization(self):
        with self._strategy.scope():
            layer = DotInteraction()
        restored = deserialize(serialize(layer))
        self.assertDictEqual(layer.get_config(), restored.get_config())

    def test_model_saving(self):
        with self._strategy.scope():
            feature1 = keras.layers.Input(shape=(5,))
            feature2 = keras.layers.Input(shape=(5,))
            feature3 = keras.layers.Input(shape=(5,))
            x = DotInteraction()([feature1, feature2, feature3])
            x = keras.layers.Dense(units=1)(x)
            model = keras.Model([feature1, feature2, feature3], x)

        self.run_model_saving_test(
            model=model,
            input_data=self.input,
        )


if __name__ == "__main__":
    absltest.main()
