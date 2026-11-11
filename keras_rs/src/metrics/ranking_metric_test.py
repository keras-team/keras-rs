from absl.testing import parameterized
from keras import ops

from keras_rs.src import testing
from keras_rs.src.metrics.dcg import DCG


class RankingMetricTest(testing.TestCase, parameterized.TestCase):
    def setUp(self):
        super().setUp()

        self.y_true_batched = ops.array(
            [[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 3.0, 2.0]], dtype="float32"
        )
        self.y_pred_batched = ops.array(
            [[0.1, 0.2, 0.9, 0.3], [0.1, 0.8, 0.9, 0.7]], dtype="float32"
        )

    @parameterized.named_parameters(
        ("1d_wrong_batch_size", [1.0, 1.0, 1.0], "one weight per list"),
        ("1d_per_item", [1.0, 1.0, 1.0, 1.0], "one weight per list"),
        ("2d_per_list", [[1.0], [2.0]], "same shape as `y_true`"),
        (
            "2d_shared_per_item",
            [[1.0, 1.0, 1.0, 1.0]],
            "same shape as `y_true`",
        ),
        (
            "2d_wrong_list_size",
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            "same shape as `y_true`",
        ),
    )
    def test_invalid_batched_sample_weight_shape(
        self, sample_weight, expected_message
    ):
        metric = DCG()
        with self.assertRaisesRegex(ValueError, expected_message):
            metric.update_state(
                self.y_true_batched,
                self.y_pred_batched,
                sample_weight=ops.array(sample_weight, dtype="float32"),
            )

    def test_invalid_unbatched_sample_weight_shape(self):
        metric = DCG()
        with self.assertRaisesRegex(ValueError, "same shape as `y_true`"):
            metric.update_state(
                self.y_true_batched[0],
                self.y_pred_batched[0],
                sample_weight=ops.array([1.0, 1.0, 1.0], dtype="float32"),
            )

    def test_invalid_sample_weight_rank(self):
        metric = DCG()
        with self.assertRaisesRegex(
            ValueError, "`sample_weight` should have a rank from"
        ):
            metric.update_state(
                self.y_true_batched,
                self.y_pred_batched,
                sample_weight=ops.ones((2, 4, 1), dtype="float32"),
            )
