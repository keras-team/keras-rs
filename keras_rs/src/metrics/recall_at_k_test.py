import keras
from absl.testing import parameterized
from keras import ops
from keras.metrics import deserialize
from keras.metrics import serialize

from keras_rs.src import testing
from keras_rs.src.metrics.recall_at_k import RecallAtK


class RecallAtKTest(testing.TestCase, parameterized.TestCase):
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
            RecallAtK(k=0)
        with self.assertRaisesRegex(
            ValueError, "`k` should be a positive integer"
        ):
            RecallAtK(k=-5)
        with self.assertRaisesRegex(
            ValueError, "`k` should be a positive integer"
        ):
            RecallAtK(k=3.5)

    @parameterized.named_parameters(
        (
            "one_relevant",
            [0.0, 0.0, 1.0, 0.0],
            [0.1, 0.2, 0.9, 0.3],
            None,
            1.0,
        ),
        (
            "two_relevant",
            [1.0, 1.0, 0.0, 0.0],
            [0.8, 0.1, 0.7, 0.2],
            None,
            0.5,
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
            [1.0, 1.0, 0.0, 0.0],
            [0.8, 0.1, 0.7, 0.2],
            5.0,
            0.5,
        ),
        (
            "sample_weight_1d",
            [1.0, 1.0, 0.0, 0.0],
            [0.8, 0.1, 0.7, 0.2],
            [2.0, 1.0, 3.0, 0.0],
            1.0,
        ),
    )
    def test_unbatched_inputs(
        self, y_true, y_pred, sample_weight, expected_output
    ):
        r_at_k = RecallAtK(k=3)
        r_at_k.update_state(y_true, y_pred, sample_weight=sample_weight)
        result = r_at_k.result()
        self.assertAllClose(result, expected_output, rtol=1e-6)

    def test_batched_input(self):
        r_at_k = RecallAtK(k=3)
        r_at_k.update_state(self.y_true_batched, self.y_pred_batched)
        result = r_at_k.result()
        self.assertAllClose(result, 0.541667)

    @parameterized.named_parameters(
        ("scalar_0.5", 0.5, 0.541667),
        ("scalar_0", 0, 0),
        ("1d", [1.0, 0.5, 2.0, 1.0], 0.55),
    )
    def test_batched_inputs_sample_weight(self, sample_weight, expected_output):
        r_at_k = RecallAtK(k=3)
        r_at_k.update_state(
            self.y_true_batched,
            self.y_pred_batched,
            sample_weight=sample_weight,
        )
        result = r_at_k.result()
        self.assertAllClose(result, expected_output, rtol=1e-6)

    @parameterized.named_parameters(
        (
            "mask_relevant_items",
            [[0.0, 1.0, 1.0, 0.0]],
            [[0.5, 0.8, 0.2, 0.1]],
            [[1.0, 0.0, 0.0, 1.0]],
            0.0,
        ),
        (
            "mask_first_relevant_item",
            [[0, 0, 1, 1]],
            [[0.8, 0.2, 0.6, 0.1]],
            [[1.0, 1.0, 0.0, 1.0]],
            0.0,
        ),
        (
            "mask_irrelevant_item",
            [[0, 1, 0, 1]],
            [[0.5, 0.8, 0.2, 0.1]],
            [[0.0, 1.0, 1.0, 1.0]],
            0.5,
        ),
        (
            "general_case",
            [[0, 1, 1, 0], [1, 0, 2, 1]],
            [[0.8, 0.7, 0.1, 0.2], [0.9, 0.1, 0.2, 0.3]],
            [[0.8, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]],
            0.75,
        ),
    )
    def test_2d_sample_weight(
        self, y_true, y_pred, sample_weight, expected_output
    ):
        r_at_k = RecallAtK(k=2)
        r_at_k.update_state(y_true, y_pred, sample_weight=sample_weight)
        result = r_at_k.result()
        self.assertAllClose(result, expected_output, rtol=1e-6)

    @parameterized.named_parameters(
        (
            "mask_relevant_items",
            {
                "labels": [[0.0, 1.0, 1.0, 0.0]],
                "mask": [[True, False, False, True]],
            },
            [[0.5, 0.8, 0.2, 0.1]],
            None,
            0.0,
        ),
        (
            "mask_first_relevant_item",
            {"labels": [[0, 0, 1, 1]], "mask": [[True, True, False, True]]},
            [[0.8, 0.2, 0.6, 0.1]],
            None,
            0.0,
        ),
        (
            "mask_irrelevant_item",
            {"labels": [[0, 1, 0, 1]], "mask": [[False, True, True, True]]},
            [[0.5, 0.8, 0.2, 0.1]],
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
            0.75,
        ),
    )
    def test_masking(self, y_true, y_pred, sample_weight, expected_output):
        r_at_k = RecallAtK(k=2)
        r_at_k.update_state(y_true, y_pred, sample_weight=sample_weight)
        result = r_at_k.result()
        self.assertAllClose(result, expected_output, rtol=1e-6)

    @parameterized.named_parameters(
        ("1", 1, 0.375),
        ("2", 2, 0.458333),
        ("3", 3, 0.541667),
        ("4", 4, 0.75),
    )
    def test_k(self, k, expected_recall):
        r_at_k = RecallAtK(k=k)
        r_at_k.update_state(self.y_true_batched, self.y_pred_batched)
        result = r_at_k.result()
        self.assertAllClose(result, expected_recall)

    def test_statefulness(self):
        r_at_k = RecallAtK(k=3)
        r_at_k.update_state(self.y_true_batched[:2], self.y_pred_batched[:2])
        result = r_at_k.result()
        self.assertAllClose(result, 0.833333, rtol=1e-6)

        r_at_k.update_state(self.y_true_batched[2:], self.y_pred_batched[2:])
        result = r_at_k.result()
        self.assertAllClose(result, 0.541667)

        r_at_k.reset_state()
        result = r_at_k.result()
        self.assertAllClose(result, 0.0)

    def test_serialization(self):
        metric = RecallAtK(k=3)
        restored = deserialize(serialize(metric))
        self.assertDictEqual(metric.get_config(), restored.get_config())

    def test_model_evaluate(self):
        inputs = keras.Input(shape=(20,), dtype="float32")
        outputs = keras.layers.Dense(5)(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)

        model.compile(
            loss=keras.losses.MeanSquaredError(),
            metrics=[RecallAtK(k=3)],
            optimizer="adam",
        )
        model.evaluate(
            x=keras.random.normal((2, 20)),
            y=keras.random.randint(
                (2, 5), minval=0, maxval=2
            ),  # Using 0/1 for y_true
            verbose=0,
        )
