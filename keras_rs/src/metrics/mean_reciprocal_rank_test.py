import keras
from absl.testing import parameterized
from keras import ops
from keras.metrics import deserialize
from keras.metrics import serialize

from keras_rs.src import testing
from keras_rs.src.metrics.mean_reciprocal_rank import MeanReciprocalRank


class MeanReciprocalRankTest(testing.TestCase, parameterized.TestCase):
    def setUp(self):
        self.y_true_batched = ops.array(
            [
                [0, 0, 1, 0],
                [0, 3, 0, 0],  # Rank 2 -> MRR = 0.5
                [0, 0, 0, 0],  # Rank N/A -> MRR = 0.0
                [1, 0, 2, 0],  # Rank 1 (first) -> MRR = 1.0
            ],
            dtype="float32",
        )
        self.y_pred_batched = ops.array(
            [
                [0.1, 0.2, 0.9, 0.3],  # MRR = 1.0
                [0.8, 0.7, 0.1, 0.2],  # MRR = 0.5
                [0.4, 0.3, 0.2, 0.1],  # MRR = 0.0
                [0.9, 0.2, 0.8, 0.3],  # MRR = 1.0
            ],
            dtype="float32",
        )

    def test_invalid_k_init(self):
        with self.assertRaisesRegex(
            ValueError, "`k` should be a positive integer"
        ):
            MeanReciprocalRank(k=0)
        with self.assertRaisesRegex(
            ValueError, "`k` should be a positive integer"
        ):
            MeanReciprocalRank(k=-5)
        with self.assertRaisesRegex(
            ValueError, "`k` should be a positive integer"
        ):
            MeanReciprocalRank(k=3.5)  # type: ignore

    @parameterized.named_parameters(
        (
            "perfect_rank",
            [0.0, 0.0, 1.0, 0.0],
            [0.1, 0.2, 0.9, 0.3],
            None,
            1.0,
        ),
        (
            "second_rank",
            [0.0, 0.0, 1.0, 0.0],
            [0.8, 0.1, 0.7, 0.2],
            None,
            1 / 2,
        ),
        (
            "third_rank",
            [0.0, 0.0, 1.0, 0.0],
            [0.4, 0.3, 0.2, 0.1],
            None,
            1 / 3,
        ),
        (
            "irrelevant",
            [0.0, 0.0, 0.0, 0.0],
            [0.1, 0.2, 0.9, 0.3],
            None,
            0.0,
        ),
        (
            "multiple_relevant_items",
            [1.0, 0.0, 1.0, 0.0],
            [0.9, 0.2, 0.8, 0.3],
            None,
            1.0,
        ),
        (
            "sample_weight_0",
            [0.0, 1.0, 0.0],
            [0.5, 0.8, 0.2],
            [0.0, 0.0, 0.0],
            0.0,
        ),
        (
            "sample_weight_scalar",
            [0.0, 0.0, 1.0, 0.0],
            [0.8, 0.1, 0.7, 0.2],
            5.0,
            1 / 2,
        ),
        (
            "sample_weight_1d",
            [1.0, 0.0, 1.0, 0.0],
            [0.9, 0.2, 0.8, 0.3],
            [2.0, 1.0, 3.0, 0.0],
            1.0,
        ),
    )
    def test_unbatched_inputs(
        self, y_true, y_pred, sample_weight, expected_output
    ):
        mrr_metric = MeanReciprocalRank()
        mrr_metric.update_state(y_true, y_pred, sample_weight=sample_weight)
        result = mrr_metric.result()
        self.assertAllClose(result, expected_output)

    def test_batched_input(self):
        mrr_metric = MeanReciprocalRank()
        mrr_metric.update_state(self.y_true_batched, self.y_pred_batched)
        result = mrr_metric.result()
        self.assertAllClose(result, 0.625)

    @parameterized.named_parameters(
        ("scalar_0.5", 0.5, 0.625),
        ("scalar_0", 0, 0),
        ("1d", [1.0, 0.5, 2.0, 1.0], 0.675),
    )
    def test_batched_inputs_sample_weight(self, sample_weight, expected_output):
        mrr_metric = MeanReciprocalRank()
        mrr_metric.update_state(
            self.y_true_batched,
            self.y_pred_batched,
            sample_weight=sample_weight,
        )
        result = mrr_metric.result()
        self.assertAllClose(result, expected_output)

    @parameterized.named_parameters(
        (
            "mask_relevant_items",
            [[0.0, 1.0, 0.0]],
            [[0.5, 0.8, 0.2]],
            [[1.0, 0.0, 1.0]],
            0.0,
        ),
        (
            "mask_first_relevant_item",
            [[1, 0, 1]],
            [[0.8, 0.2, 0.6]],
            [[0.0, 1.0, 1.0]],
            1.0,
        ),
        (
            "mask_irrelevant_item",
            [[0, 1, 0]],
            [[0.5, 0.8, 0.2]],
            [[0.0, 1.0, 1.0]],
            1.0,
        ),
        (
            "general_case",
            [[0, 1, 0, 0], [1, 0, 0, 1]],
            [[0.8, 0.7, 0.1, 0.2], [0.9, 0.1, 0.2, 0.3]],
            [[0.8, 0.8, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]],
            0.777778,
        ),
    )
    def test_2d_sample_weight(
        self, y_true, y_pred, sample_weight, expected_output
    ):
        mrr_metric = MeanReciprocalRank()

        mrr_metric.update_state(y_true, y_pred, sample_weight=sample_weight)
        result = mrr_metric.result()
        self.assertAllClose(result, expected_output)

    @parameterized.named_parameters(
        (
            "mask_relevant_items",
            {"labels": [[0.0, 1.0, 0.0]], "mask": [[True, False, True]]},
            [[0.5, 0.8, 0.2]],
            None,
            0.0,
        ),
        (
            "mask_first_relevant_item",
            {"labels": [[1, 0, 1]], "mask": [[False, True, True]]},
            [[0.8, 0.2, 0.6]],
            None,
            1.0,
        ),
        (
            "mask_irrelevant_item",
            {"labels": [[0, 1, 0]], "mask": [[False, True, True]]},
            [[0.5, 0.8, 0.2]],
            None,
            1.0,
        ),
        (
            "general_case",
            {
                "labels": [[0, 1, 0, 0], [1, 0, 0, 1]],
                "mask": [
                    [True, True, False, False],
                    [False, False, True, True],
                ],
            },
            [[0.8, 0.7, 0.1, 0.2], [0.9, 0.1, 0.2, 0.3]],
            [[0.8, 0.8, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
            0.777778,
        ),
    )
    def test_masking(self, y_true, y_pred, sample_weight, expected_output):
        mrr_metric = MeanReciprocalRank()

        mrr_metric.update_state(y_true, y_pred, sample_weight=sample_weight)
        result = mrr_metric.result()
        self.assertAllClose(result, expected_output)

    @parameterized.named_parameters(
        ("1", 1, 0.5), ("2", 2, 0.625), ("3", 3, 0.625), ("4", 4, 0.625)
    )
    def test_k(self, k, expected_mrr):
        mrr_metric = MeanReciprocalRank(k=k)
        mrr_metric.update_state(self.y_true_batched, self.y_pred_batched)
        result = mrr_metric.result()
        self.assertAllClose(result, expected_mrr)

    def test_statefulness(self):
        mrr_metric = MeanReciprocalRank()
        # Batch 1: First two lists
        mrr_metric.update_state(
            self.y_true_batched[:2], self.y_pred_batched[:2]
        )
        result = mrr_metric.result()
        self.assertAllClose(result, 0.75)

        # Batch 2: Last two lists
        mrr_metric.update_state(
            self.y_true_batched[2:], self.y_pred_batched[2:]
        )
        result = mrr_metric.result()
        self.assertAllClose(result, 0.625)

        # Reset state
        mrr_metric.reset_state()
        result = mrr_metric.result()
        self.assertAllClose(result, 0.0)

    def test_serialization(self):
        metric = MeanReciprocalRank()
        restored = deserialize(serialize(metric))
        self.assertDictEqual(metric.get_config(), restored.get_config())

    def test_model_evaluate(self):
        inputs = keras.Input(shape=(20,), dtype="float32")
        outputs = keras.layers.Dense(5)(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)

        model.compile(
            loss=keras.losses.MeanSquaredError(),
            metrics=[MeanReciprocalRank()],
            optimizer="adam",
        )
        model.evaluate(
            x=keras.random.normal((2, 20)),
            y=keras.random.randint((2, 5), minval=0, maxval=4),
        )
