import keras
from absl.testing import parameterized
from keras import ops
from keras.metrics import deserialize
from keras.metrics import serialize

from keras_rs.src import testing
from keras_rs.src.metrics.precision_at_k import PrecisionAtK


class PrecisionAtKTest(testing.TestCase, parameterized.TestCase):
    def setUp(self):
        super().setUp()

        self.y_true_batched = ops.array(
            [
                [0, 0, 1, 0],
                [0, 1, 1, 1],
                [0, 0, 0, 0],
                [1, 0, 1, 0],
            ],
            dtype="float32",
        )
        self.y_pred_batched = ops.array(
            [
                [0.1, 0.2, 0.9, 0.3],
                [0.8, 0.7, 0.1, 0.2],
                [0.4, 0.3, 0.2, 0.1],
                [0.9, 0.2, 0.1, 0.3],
            ],
            dtype="float32",
        )

    def test_invalid_k_init(self):
        with self.assertRaisesRegex(
            ValueError, "`k` should be a positive integer"
        ):
            PrecisionAtK(k=0)
        with self.assertRaisesRegex(
            ValueError, "`k` should be a positive integer"
        ):
            PrecisionAtK(k=-5)
        with self.assertRaisesRegex(
            ValueError, "`k` should be a positive integer"
        ):
            PrecisionAtK(k=3.5)

    @parameterized.named_parameters(
        (
            "one_relevant",
            [0.0, 0.0, 1.0, 0.0],
            [0.1, 0.2, 0.9, 0.3],
            None,
            1 / 3,
        ),
        (
            "two_relevant",
            [1.0, 0.0, 1.0, 0.0],
            [0.8, 0.1, 0.7, 0.2],
            None,
            2 / 3,
        ),
        (
            "irrelevant",
            [0.0, 0.0, 0.0, 0.0],
            [0.1, 0.2, 0.9, 0.3],
            None,
            0.0,
        ),
        (
            "sample_weight_0",
            [1.0, 1.0, 0.0],
            [0.5, 0.8, 0.2],
            [0.0, 0.0, 0.0],
            0.0,
        ),
        (
            "sample_weight_scalar",
            [1.0, 0.0, 1.0, 0.0],
            [0.8, 0.1, 0.7, 0.2],
            5.0,
            2 / 3,
        ),
        (
            "sample_weight_1d",
            [1.0, 0.0, 1.0, 0.0],
            [0.8, 0.1, 0.7, 0.2],
            [2.0, 1.0, 3.0, 0.0],
            2 / 3,
        ),
    )
    def test_unbatched_inputs(
        self, y_true, y_pred, sample_weight, expected_output
    ):
        p_at_k = PrecisionAtK(k=3)
        p_at_k.update_state(y_true, y_pred, sample_weight=sample_weight)
        result = p_at_k.result()
        self.assertAllClose(result, expected_output, rtol=1e-6)

    def test_batched_input(self):
        p_at_k = PrecisionAtK(k=3)
        p_at_k.update_state(self.y_true_batched, self.y_pred_batched)
        result = p_at_k.result()
        self.assertAllClose(result, 1 / 3)

    @parameterized.named_parameters(
        ("scalar_0.5", 0.5, 1 / 3),
        ("scalar_0", 0, 0),
        ("1d", [1.0, 0.5, 2.0, 1.0], 0.3),
    )
    def test_batched_inputs_sample_weight(self, sample_weight, expected_output):
        p_at_k = PrecisionAtK(k=3)
        p_at_k.update_state(
            self.y_true_batched,
            self.y_pred_batched,
            sample_weight=sample_weight,
        )
        result = p_at_k.result()
        self.assertAllClose(result, expected_output, rtol=1e-6)

    @parameterized.named_parameters(
        (
            "mask_relevant_items",
            [[0.0, 1.0, 1.0]],
            [[0.5, 0.8, 0.2]],
            [[1.0, 0.0, 0.0]],
            0.0,
        ),
        (
            "mask_first_relevant_item",
            [[1, 0, 1]],
            [[0.8, 0.2, 0.6]],
            [[0.0, 1.0, 1.0]],
            0.5,
        ),
        (
            "mask_irrelevant_item",
            [[0, 1, 0]],
            [[0.5, 0.8, 0.2]],
            [[0.0, 1.0, 1.0]],
            0.5,
        ),
        (
            "general_case",
            [[0, 1, 1, 0], [1, 0, 2, 1]],
            [[0.8, 0.7, 0.1, 0.2], [0.9, 0.1, 0.2, 0.3]],
            [[0.8, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]],
            0.583333,
        ),
    )
    def test_2d_sample_weight(
        self, y_true, y_pred, sample_weight, expected_output
    ):
        p_at_k = PrecisionAtK(k=3)
        p_at_k.update_state(y_true, y_pred, sample_weight=sample_weight)
        result = p_at_k.result()
        self.assertAllClose(result, expected_output, rtol=1e-6)

    @parameterized.named_parameters(
        (
            "mask_relevant_items",
            {"labels": [[0.0, 1.0, 1.0]], "mask": [[True, False, False]]},
            [[0.5, 0.8, 0.2]],
            None,
            0.0,
        ),
        (
            "mask_first_relevant_item",
            {"labels": [[1, 0, 1]], "mask": [[False, True, True]]},
            [[0.8, 0.2, 0.6]],
            None,
            0.5,
        ),
        (
            "mask_irrelevant_item",
            {"labels": [[0, 1, 0]], "mask": [[False, True, True]]},
            [[0.5, 0.8, 0.2]],
            None,
            0.5,
        ),
        (
            "general_case",
            {
                "labels": [[0, 1, 1, 0], [1, 0, 2, 1]],
                "mask": [
                    [True, True, True, False],
                    [True, True, False, False],
                ],
            },
            [[0.8, 0.7, 0.1, 0.2], [0.9, 0.1, 0.2, 0.3]],
            [[0.8, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]],
            0.583333,
        ),
    )
    def test_masking(self, y_true, y_pred, sample_weight, expected_output):
        p_at_k = PrecisionAtK(k=3)
        p_at_k.update_state(y_true, y_pred, sample_weight=sample_weight)
        result = p_at_k.result()
        self.assertAllClose(result, expected_output, rtol=1e-6)

    @parameterized.named_parameters(
        ("1", 1, 0.5),
        ("2", 2, 0.375),
        ("3", 3, 0.333333),
        ("4", 4, 0.375),
    )
    def test_k(self, k, expected_precision):
        p_at_k = PrecisionAtK(k=k)
        p_at_k.update_state(self.y_true_batched, self.y_pred_batched)
        result = p_at_k.result()
        self.assertAllClose(result, expected_precision)

    def test_statefulness(self):
        p_at_k = PrecisionAtK(k=3)
        p_at_k.update_state(self.y_true_batched[:2], self.y_pred_batched[:2])
        result = p_at_k.result()
        self.assertAllClose(result, 0.5, rtol=1e-6)

        p_at_k.update_state(self.y_true_batched[2:], self.y_pred_batched[2:])
        result = p_at_k.result()
        self.assertAllClose(result, 1 / 3)

        p_at_k.reset_state()
        result = p_at_k.result()
        self.assertAllClose(result, 0.0)

    def test_serialization(self):
        metric = PrecisionAtK(k=3)
        restored = deserialize(serialize(metric))
        self.assertDictEqual(metric.get_config(), restored.get_config())

    def test_model_evaluate(self):
        inputs = keras.Input(shape=(20,), dtype="float32")
        outputs = keras.layers.Dense(5)(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)

        model.compile(
            loss=keras.losses.MeanSquaredError(),
            metrics=[PrecisionAtK(k=3)],
            optimizer="adam",
        )
        model.evaluate(
            x=keras.random.normal((2, 20)),
            y=keras.random.randint(
                (2, 5), minval=0, maxval=2
            ),  # Using 0/1 for y_true
            verbose=0,
        )
