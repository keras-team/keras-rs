import keras
from absl.testing import parameterized
from keras import ops
from keras.layers import deserialize
from keras.layers import serialize

from keras_rs.src import testing
from keras_rs.src.layers.modeling.dot_interaction import DotInteraction


class DotInteractionTest(testing.TestCase, parameterized.TestCase):
    def setUp(self):
        self.input = [
            ops.array([[0.1, -4.3, 0.2, 1.1, 0.3]]),
            ops.array([[2.0, 3.2, -1.0, 0.0, 1.0]]),
            ops.array([[0.0, 1.0, -3.0, -2.2, -0.2]]),
        ]

        feature1 = self.input[0]
        feature2 = self.input[1]
        feature3 = self.input[2]

        self.f11 = ops.dot(feature1[0], feature1[0])
        self.f12 = ops.dot(feature1[0], feature2[0])
        self.f13 = ops.dot(feature1[0], feature3[0])
        self.f22 = ops.dot(feature2[0], feature2[0])
        self.f23 = ops.dot(feature2[0], feature3[0])
        self.f33 = ops.dot(feature3[0], feature3[0])

    def test_layer_call_1(self):
        layer = DotInteraction(self_interaction=False, skip_gather=False)
        output = layer(self.input)
        print(f"{output=}")
        print(ops.array([[self.f12, self.f13, self.f23]]))
        self.assertAllClose(output, ops.array([[self.f12, self.f13, self.f23]]))

        # def test_layer_call_2(self):
        layer = DotInteraction(self_interaction=False, skip_gather=True)
        output = layer(self.input)

        # Test output.
        self.assertAllClose(
            output,
            ops.array([[0, 0, 0, self.f12, 0, 0, self.f13, self.f23, 0]]),
        )

    def test_layer_call_3(self):
        layer = DotInteraction(self_interaction=True, skip_gather=False)
        output = layer(self.input)

        # Test output.
        self.assertAllClose(
            output,
            ops.array(
                [[self.f11, self.f12, self.f22, self.f13, self.f23, self.f33]]
            ),
        )

    def test_layer_call_4(self):
        layer = DotInteraction(self_interaction=True, skip_gather=True)
        output = layer(self.input)

        # Test output.
        self.assertAllClose(
            output,
            ops.array(
                [
                    [
                        self.f11,
                        0,
                        0,
                        self.f12,
                        self.f22,
                        0,
                        self.f13,
                        self.f23,
                        self.f33,
                    ]
                ]
            ),
        )

    def test_serialization(self):
        sampler = DotInteraction()
        restored = deserialize(serialize(sampler))
        self.assertDictEqual(sampler.get_config(), restored.get_config())

    def test_model_saving(self):
        def get_model():
            feature1 = keras.layers.Input(shape=(5,))
            feature2 = keras.layers.Input(shape=(5,))
            feature3 = keras.layers.Input(shape=(5,))
            x = DotInteraction()([feature1, feature2, feature3])
            x = keras.layers.Dense(units=1)(x)
            model = keras.Model([feature1, feature2, feature3], x)
            return model

        self.run_model_saving_test(
            model=get_model(),
            input_data=self.input,
        )
