import keras
from absl.testing import parameterized
from keras import ops
from keras.layers import deserialize
from keras.layers import serialize

from keras_rs.src import testing
from keras_rs.src.layers.feature_interaction.feature_cross import FeatureCross


class FeatureCrossTest(testing.TestCase, parameterized.TestCase):
    def setUp(self):
        self.x0 = ops.array([[0.1, 0.2, 0.3]], dtype="float32")
        self.x = ops.array([[0.4, 0.5, 0.6]], dtype="float32")
        self.exp_output = ops.array([[0.55, 0.8, 1.05]])

        self.one_inp_exp_output = ops.array([[0.16, 0.32, 0.48]])

    def test_full_layer(self):
        layer = FeatureCross(projection_dim=None, kernel_initializer="ones")
        output = layer(self.x0, self.x)

        # Test output.
        self.assertAllClose(self.exp_output, output)

        # Test which layers have been initialised and their shapes.
        # Kernel, bias terms corresponding to dense layer.
        self.assertLen(layer.weights, 2, msg="Unexpected number of `weights`")
        self.assertEqual(layer.weights[0].shape, (3, 3))
        self.assertEqual(layer.weights[1].shape, (3,))

    def test_low_rank_layer(self):
        layer = FeatureCross(projection_dim=1, kernel_initializer="ones")
        output = layer(self.x0, self.x)

        # Test output.
        self.assertAllClose(self.exp_output, output)

        # Test which layers have been initialised and their shapes.
        # Kernel term corresponding to down projection layer, and kernel,
        # bias terms corresponding to dense layer.
        self.assertLen(layer.weights, 3, msg="Unexpected number of `weights`")
        self.assertEqual(layer.weights[0].shape, (3, 1))
        self.assertEqual(layer.weights[1].shape, (1, 3))
        self.assertEqual(layer.weights[2].shape, (3,))

    def test_one_input(self):
        layer = FeatureCross(projection_dim=None, kernel_initializer="ones")
        output = layer(self.x0)
        self.assertAllClose(self.one_inp_exp_output, output)

    def test_invalid_input_shapes(self):
        x0 = ops.ones((12, 5))
        x = ops.ones((12, 7))

        layer = FeatureCross()

        with self.assertRaises(ValueError):
            layer(x0, x)

    def test_invalid_diag_scale(self):
        with self.assertRaises(ValueError):
            FeatureCross(diag_scale=-1.0)

    def test_diag_scale(self):
        layer = FeatureCross(
            projection_dim=None, diag_scale=1.0, kernel_initializer="ones"
        )
        output = layer(self.x0, self.x)

        self.assertAllClose(ops.array([[0.59, 0.9, 1.23]]), output)

    def test_pre_activation(self):
        layer = FeatureCross(projection_dim=None, pre_activation=ops.zeros_like)
        output = layer(self.x0, self.x)

        self.assertAllClose(self.x, output)

    def test_predict(self):
        x0 = keras.layers.Input(shape=(3,))
        x1 = FeatureCross(projection_dim=None)(x0, x0)
        x2 = FeatureCross(projection_dim=None)(x0, x1)
        logits = keras.layers.Dense(units=1)(x2)
        model = keras.Model(x0, logits)

        model.predict(self.x0, batch_size=2)

    def test_serialization(self):
        sampler = FeatureCross(projection_dim=None, pre_activation="swish")
        restored = deserialize(serialize(sampler))
        self.assertDictEqual(sampler.get_config(), restored.get_config())

    def test_model_saving(self):
        x0 = keras.layers.Input(shape=(3,))
        x1 = FeatureCross(projection_dim=None)(x0, x0)
        x2 = FeatureCross(projection_dim=None)(x0, x1)
        logits = keras.layers.Dense(units=1)(x2)
        model = keras.Model(x0, logits)

        self.run_model_saving_test(
            model=model,
            input_data=self.x0,
        )
