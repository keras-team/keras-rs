from absl.testing import parameterized
from keras import ops
from keras.metrics import deserialize
from keras.metrics import serialize

from keras_rs.src import testing
from keras_rs.src.metrics.mean_average_precision import MeanAveragePrecision


class MeanAveragePrecisionTest(testing.TestCase, parameterized.TestCase):
    def setUp(self):
        # Unbatched inputs
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
        self.y_true_unbatched_multi = ops.array([1, 0, 2, 0], dtype="float32")
        self.y_pred_unbatched_multi = ops.array(
            [0.9, 0.2, 0.8, 0.3], dtype="float32"
        )

        # Batched inputs
        self.y_true_batched = ops.array(
            [
                [0, 0, 1, 0],
                [0, 3, 0, 0],
                [0, 0, 0, 0],
                [1, 0, 2, 0],
            ],
            dtype="float32",
        )
        self.y_pred_batched = ops.array(
            [
                [0.1, 0.2, 0.9, 0.3],
                [0.8, 0.7, 0.1, 0.2],
                [0.4, 0.3, 0.2, 0.1],
                [0.9, 0.2, 0.8, 0.3],
            ],
            dtype="float32",
        )

    def test_invalid_k_init(self):
        with self.assertRaisesRegex(
            ValueError, "`k` should be a positive integer"
        ):
            MeanAveragePrecision(k=0)
        with self.assertRaisesRegex(
            ValueError, "`k` should be a positive integer"
        ):
            MeanAveragePrecision(k=-5)
        with self.assertRaisesRegex(
            ValueError, "`k` should be a positive integer"
        ):
            MeanAveragePrecision(k=3.5)

    def test_unbatched_perfect_rank(self):
        map_metric = MeanAveragePrecision()
        map_metric.update_state(
            self.y_true_unbatched, self.y_pred_unbatched_perfect
        )
        result = map_metric.result()
        self.assertAllClose(result, 1.0)

    def test_unbatched_second_rank(self):
        map_metric = MeanAveragePrecision()
        map_metric.update_state(
            self.y_true_unbatched, self.y_pred_unbatched_second
        )
        result = map_metric.result()
        self.assertAllClose(result, 0.5)

    def test_unbatched_third_rank(self):
        map_metric = MeanAveragePrecision()
        map_metric.update_state(
            self.y_true_unbatched, self.y_pred_unbatched_third
        )
        result = map_metric.result()
        self.assertAllClose(result, 1 / 3)

    def test_unbatched_no_relevant(self):
        map_metric = MeanAveragePrecision()
        map_metric.update_state(
            self.y_true_unbatched_none,
            self.y_pred_unbatched_perfect,
        )
        result = map_metric.result()
        self.assertAllClose(result, 0.0)

    def test_unbatched_multiple_relevant(self):
        map_metric = MeanAveragePrecision()
        map_metric.update_state(
            self.y_true_unbatched_multi, self.y_pred_unbatched_multi
        )
        result = map_metric.result()
        self.assertAllClose(result, 1.0)

    def test_batched_input(self):
        map_metric = MeanAveragePrecision()
        map_metric.update_state(self.y_true_batched, self.y_pred_batched)
        result = map_metric.result()
        self.assertAllClose(result, 0.625)

    @parameterized.named_parameters(
        ("1", 1, 0.375),
        ("2", 2, 0.625),
        ("3", 3, 0.625),
        ("4", 4, 0.625),
    )
    def test_k(self, k, expected_map):
        map_metric = MeanAveragePrecision(k=k)
        map_metric.update_state(self.y_true_batched, self.y_pred_batched)
        result = map_metric.result()
        self.assertAllClose(result, expected_map)

    def test_statefulness(self):
        map_metric = MeanAveragePrecision()
        # Batch 1: First two lists
        map_metric.update_state(
            self.y_true_batched[:2], self.y_pred_batched[:2]
        )
        result1 = map_metric.result()
        self.assertAllClose(result1, 0.75)

        # Batch 2: Last two lists
        map_metric.update_state(
            self.y_true_batched[2:], self.y_pred_batched[2:]
        )
        result2 = map_metric.result()
        self.assertAllClose(result2, 0.625)

        # Reset state
        map_metric.reset_state()
        result3 = map_metric.result()
        self.assertAllClose(result3, 0.0)

    @parameterized.named_parameters(
        (
            "weight_0.5",
            0.5,
            0.625,
        ),
        ("weight_0", 0.0, 0.0),
    )
    def test_scalar_sample_weight(self, sample_weight, expected_output):
        map_metric = MeanAveragePrecision()
        map_metric.update_state(
            self.y_true_batched,
            self.y_pred_batched,
            sample_weight=sample_weight,
        )
        result = map_metric.result()
        self.assertAllClose(result, expected_output)

    def test_1d_sample_weight(self):
        map_metric = MeanAveragePrecision()
        sample_weight = ops.array([1.0, 0.5, 2.0, 1.0], dtype="float32")
        map_metric.update_state(
            self.y_true_batched,
            self.y_pred_batched,
            sample_weight=sample_weight,
        )
        result = map_metric.result()
        self.assertAllClose(result, 0.703125)

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
            "batch_size_2_masking",
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
                [
                    [0.8, 0.8, 0.8, 0.8],
                    [0.0, 0.0, 1.0, 1.0],
                ],
                dtype="float32",
            ),
            0.777778,
        ),
    )
    def test_item_sample_weight(
        self, y_true, y_pred, sample_weight, expected_output
    ):
        map_metric = MeanAveragePrecision()

        map_metric.update_state(y_true, y_pred, sample_weight=sample_weight)
        result = map_metric.result()
        self.assertAllClose(result, expected_output)

    def test_serialization(self):
        metric = MeanAveragePrecision(k=10, name="map_test")
        config = serialize(metric)
        restored = deserialize(config)
        self.assertIsInstance(restored, MeanAveragePrecision)
        self.assertEqual(metric.k, restored.k)
        self.assertEqual(metric.name, restored.name)
        self.assertEqual(metric.dtype, restored.dtype)
