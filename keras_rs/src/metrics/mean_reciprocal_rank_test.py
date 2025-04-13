import keras
from absl.testing import parameterized
from keras import ops
from keras.metrics import deserialize
from keras.metrics import serialize

from keras_rs.src import testing
from keras_rs.src.metrics.mean_reciprocal_rank import MeanReciprocalRank


class MeanReciprocalRankTest(testing.TestCase, parameterized.TestCase):
    def setUp(self):
        # === Unbatched inputs ===
        self.y_true_unbatched = ops.array([0, 0, 1, 0], dtype="float32")
        self.y_pred_unbatched_perfect = ops.array(
            [0.1, 0.2, 0.9, 0.3], dtype="float32"
        )
        self.y_pred_unbatched_second = ops.array(
            [0.8, 0.1, 0.7, 0.2], dtype="float32"
        )
        self.y_pred_unbatched_third = ops.array(
            [0.4, 0.3, 0.2, 0.1], dtype="float32"
        )
        self.y_true_unbatched_none = ops.array([0, 0, 0, 0], dtype="float32")
        self.y_true_unbatched_multi = ops.array([1, 0, 1, 0], dtype="float32")
        self.y_pred_unbatched_multi = ops.array(
            [0.9, 0.2, 0.8, 0.3], dtype="float32"
        )

        # === Batched inputs ===
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

    def test_unbatched_perfect_rank(self):
        mrr_metric = MeanReciprocalRank()
        mrr_metric.update_state(
            self.y_true_unbatched, self.y_pred_unbatched_perfect
        )
        result = mrr_metric.result()
        self.assertAllClose(result, 1.0)

    def test_unbatched_second_rank(self):
        mrr_metric = MeanReciprocalRank()
        mrr_metric.update_state(
            self.y_true_unbatched, self.y_pred_unbatched_second
        )
        result = mrr_metric.result()
        self.assertAllClose(result, 0.5)

    def test_unbatched_third_rank(self):
        mrr_metric = MeanReciprocalRank()
        mrr_metric.update_state(
            self.y_true_unbatched, self.y_pred_unbatched_third
        )
        result = mrr_metric.result()
        self.assertAllClose(result, 1 / 3)

    def test_unbatched_no_relevant(self):
        mrr_metric = MeanReciprocalRank()
        mrr_metric.update_state(
            self.y_true_unbatched_none, self.y_pred_unbatched_perfect
        )
        result = mrr_metric.result()
        self.assertAllClose(result, 0.0)

    def test_unbatched_multiple_relevant(self):
        mrr_metric = MeanReciprocalRank()
        mrr_metric.update_state(
            self.y_true_unbatched_multi, self.y_pred_unbatched_multi
        )
        result = mrr_metric.result()
        self.assertAllClose(result, 1.0)

    def test_batched_input(self):
        mrr_metric = MeanReciprocalRank()
        mrr_metric.update_state(self.y_true_batched, self.y_pred_batched)
        result = mrr_metric.result()
        self.assertAllClose(result, 0.625)

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
        result1 = mrr_metric.result()
        self.assertAllClose(result1, 0.75)

        # Batch 2: Last two lists
        mrr_metric.update_state(
            self.y_true_batched[2:], self.y_pred_batched[2:]
        )
        result2 = mrr_metric.result()
        self.assertAllClose(result2, 0.625)

        # Reset state
        mrr_metric.reset_state()
        result3 = mrr_metric.result()
        self.assertAllClose(result3, 0.0)

    @parameterized.named_parameters(
        ("0.5", 0.5, 0.625),
        ("0", 0, 0),
    )
    def test_scalar_sample_weight(self, sample_weight, expected_output):
        mrr_metric = MeanReciprocalRank()
        mrr_metric.update_state(
            self.y_true_batched,
            self.y_pred_batched,
            sample_weight=sample_weight,
        )
        result = mrr_metric.result()
        self.assertAllClose(result, expected_output)

    def test_1d_sample_weight(self):
        mrr_metric = MeanReciprocalRank()
        sample_weight = ops.array([1.0, 0.5, 2.0, 1.0], dtype="float32")
        mrr_metric.update_state(
            self.y_true_batched,
            self.y_pred_batched,
            sample_weight=sample_weight,
        )
        result = mrr_metric.result()
        self.assertAllClose(result, 0.675)

    @parameterized.named_parameters(
        (
            "mask_relevant",
            ops.array([[0, 1, 0]], dtype="float32"),
            ops.array([[0.5, 0.8, 0.2]], dtype="float32"),
            ops.array([[1.0, 0.0, 1.0]], dtype="float32"),
            0.0,
        ),
        (
            "mask_first_relevant",
            ops.array([[1, 0, 1]], dtype="float32"),
            ops.array([[0.8, 0.2, 0.6]], dtype="float32"),
            ops.array([[0.0, 1.0, 1.0]], dtype="float32"),
            1.0,
        ),
        (
            "mask_irrelevant",
            ops.array([[0, 1, 0]], dtype="float32"),
            ops.array([[0.5, 0.8, 0.2]], dtype="float32"),
            ops.array([[0.0, 1.0, 1.0]], dtype="float32"),
            1.0,
        ),
        (
            "batch_size_2",
            ops.array(
                [
                    [0, 1, 0, 0],
                    [1, 0, 0, 1],
                ],
                dtype="float32",
            ),
            ops.array(
                [
                    [0.8, 0.7, 0.1, 0.2],
                    [0.9, 0.1, 0.2, 0.3],
                ],
                dtype="float32",
            ),
            ops.array(
                [[0.8, 0.8, 0.8, 0.8], [0.0, 0.0, 1.0, 1.0]], dtype="float32"
            ),
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
