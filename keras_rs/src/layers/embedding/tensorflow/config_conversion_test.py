import keras
import pytest
import tensorflow as tf
from absl.testing import parameterized

from keras_rs.src import testing
from keras_rs.src.layers.embedding import distributed_embedding_config as config
from keras_rs.src.layers.embedding.tensorflow import config_conversion


@pytest.mark.skipif(
    keras.backend.backend() != "tensorflow",
    reason="Backend specific test",
)
class ConfigConversionTest(testing.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        (
            "sgd_str",
            "sgd",
            tf.tpu.experimental.embedding.SGD,
            {},
        ),
        (
            "sgd_keras",
            keras.optimizers.SGD(learning_rate=0.1),
            tf.tpu.experimental.embedding.SGD,
            {"learning_rate": 0.1},
        ),
        (
            "adagrad_keras",
            keras.optimizers.Adagrad(
                learning_rate=0.2, initial_accumulator_value=0.1
            ),
            tf.tpu.experimental.embedding.Adagrad,
            {"learning_rate": 0.2, "initial_accumulator_value": 0.1},
        ),
        (
            "adam_keras",
            keras.optimizers.Adam(
                learning_rate=0.3, beta_1=0.8, beta_2=0.9, epsilon=1e-6
            ),
            tf.tpu.experimental.embedding.Adam,
            {
                "learning_rate": 0.3,
                "beta_1": 0.8,
                "beta_2": 0.9,
                "epsilon": 1e-6,
            },
        ),
        (
            "ftrl_keras",
            keras.optimizers.Ftrl(
                learning_rate=0.4,
                learning_rate_power=-0.6,
                initial_accumulator_value=0.2,
                l1_regularization_strength=0.01,
                l2_regularization_strength=0.02,
                beta=0.9,
            ),
            tf.tpu.experimental.embedding.FTRL,
            {
                "learning_rate": 0.4,
                "learning_rate_power": -0.6,
                "initial_accumulator_value": 0.2,
                "l1_regularization_strength": 0.01,
                "l2_regularization_strength": 0.02,
                "beta": 0.9,
            },
        ),
    )
    def test_optimizer_conversion(
        self, optimizer, expected_cls, expected_params
    ):
        tpu_optimizer = config_conversion.to_tf_tpu_optimizer(optimizer)
        self.assertIsInstance(tpu_optimizer, expected_cls)
        for key, value in expected_params.items():
            self.assertAllClose(getattr(tpu_optimizer, key), value)

    def test_optimizer_conversion_with_schedule(self):
        schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-2, decay_steps=10000, decay_rate=0.9
        )
        optimizer = keras.optimizers.SGD(learning_rate=schedule)
        tpu_optimizer = config_conversion.to_tf_tpu_optimizer(optimizer)

        self.assertIsInstance(tpu_optimizer, tf.tpu.experimental.embedding.SGD)
        for step in range(1, 5):
            lr = schedule(step)
            self.assertAllClose(tpu_optimizer.learning_rate(), lr)

    @parameterized.named_parameters(
        ("unsupported_optimizer", keras.optimizers.RMSprop()),
        ("unsupported_option", keras.optimizers.SGD(momentum=0.1)),
    )
    def test_optimizer_conversion_unsupported(self, optimizer):
        with self.assertRaises(ValueError):
            config_conversion.to_tf_tpu_optimizer(optimizer)

    def test_table_conversion(self):
        table_config = config.TableConfig(
            name="table_a",
            vocabulary_size=100,
            embedding_dim=16,
            initializer="uniform",
            optimizer=keras.optimizers.SGD(learning_rate=0.1),
            combiner="mean",
        )
        tf_table_config = config_conversion.keras_to_tf_tpu_table_config(
            table_config
        )

        self.assertEqual(tf_table_config.name, table_config.name)
        self.assertEqual(
            tf_table_config.vocabulary_size, table_config.vocabulary_size
        )
        self.assertEqual(tf_table_config.dim, table_config.embedding_dim)
        self.assertEqual(tf_table_config.combiner, table_config.combiner)
        self.assertIsInstance(
            tf_table_config.initializer, keras.initializers.RandomUniform
        )
        self.assertIsInstance(
            tf_table_config.optimizer, tf.tpu.experimental.embedding.SGD
        )
        self.assertAllClose(tf_table_config.optimizer.learning_rate, 0.1)

    def test_feature_and_config_conversion(self):
        table_a = config.TableConfig(
            name="table_a",
            vocabulary_size=100,
            embedding_dim=16,
            optimizer=keras.optimizers.SGD(learning_rate=0.1),
        )
        table_b = config.TableConfig(
            name="table_b",
            vocabulary_size=200,
            embedding_dim=32,
            initializer=keras.initializers.RandomUniform(),
        )
        feature_configs = {
            "feature_a": config.FeatureConfig(
                name="feature_a",
                table=table_a,
                input_shape=(64, 1),
                output_shape=(64, 16),
            ),
            "feature_b": config.FeatureConfig(
                name="feature_b",
                table=table_b,
                input_shape=(64, 1),
                output_shape=(64, 32),
            ),
            "feature_c": config.FeatureConfig(
                name="feature_c",
                table=table_b,
                input_shape=(64, 1),
                output_shape=(64, 32),
            ),
        }
        num_replicas_in_sync = 8
        expected_output_shape = (64 // num_replicas_in_sync,)

        (
            tf_feature_configs,
            sparse_core_config,
        ) = config_conversion.keras_to_tf_tpu_configuration(
            feature_configs,
            table_stacking="auto",
            num_replicas_in_sync=num_replicas_in_sync,
        )

        self.assertIsInstance(tf_feature_configs, dict)
        self.assertLen(tf_feature_configs, 3)

        # Check feature_a
        tf_feature_a = tf_feature_configs["feature_a"]
        self.assertEqual(tf_feature_a.name, "feature_a")
        self.assertEqual(
            tuple(tf_feature_a.output_shape), expected_output_shape
        )
        self.assertEqual(tf_feature_a.table.name, "table_a")
        self.assertEqual(tf_feature_a.table.vocabulary_size, 100)
        self.assertEqual(tf_feature_a.table.dim, 16)
        self.assertIsInstance(
            tf_feature_a.table.optimizer, tf.tpu.experimental.embedding.SGD
        )
        self.assertAllClose(tf_feature_a.table.optimizer.learning_rate, 0.1)

        # Check feature_b
        tf_feature_b = tf_feature_configs["feature_b"]
        self.assertEqual(tf_feature_b.name, "feature_b")
        self.assertEqual(
            tuple(tf_feature_b.output_shape), expected_output_shape
        )
        self.assertEqual(tf_feature_b.table.name, "table_b")
        self.assertEqual(tf_feature_b.table.vocabulary_size, 200)
        self.assertEqual(tf_feature_b.table.dim, 32)
        self.assertIs(tf_feature_b.table.initializer, table_b.initializer)

        # Check feature_c
        tf_feature_c = tf_feature_configs["feature_c"]
        self.assertEqual(tf_feature_c.name, "feature_c")
        self.assertEqual(
            tuple(tf_feature_c.output_shape), expected_output_shape
        )
        self.assertIs(tf_feature_c.table, tf_feature_b.table)


if __name__ == "__main__":
    testing.main()
