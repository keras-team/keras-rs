import math

from absl.testing import parameterized
from keras import ops
from keras.metrics import deserialize
from keras.metrics import serialize

from keras_rs.src import testing
from keras_rs.src.metrics.dcg import DCG


class DCGTest(testing.TestCase, parameterized.TestCase):
    def setUp(self):
        # === Unbatched Inputs ===
        # Binary relevance example
        self.y_true_unbatched = ops.array([0, 0, 1, 0], dtype="float32")
        self.y_pred_unbatched_perfect = ops.array(
            [0.1, 0.2, 0.9, 0.3], dtype="float32"
        )
        self.y_pred_unbatched_good = ops.array(
            [0.8, 0.1, 0.7, 0.2], dtype="float32"
        )
        self.y_pred_unbatched_bad = ops.array(
            [0.4, 0.3, 0.2, 0.1], dtype="float32"
        )

        # No relevant items
        self.y_true_unbatched_none = ops.array([0, 0, 0, 0], dtype="float32")

        # Graded relevance example
        self.y_true_unbatched_graded = ops.array([1, 0, 3, 2], dtype="float32")
        self.y_pred_unbatched_graded_perfect = ops.array(
            [0.1, 0.8, 0.9, 0.7], dtype="float32"
        )
        self.y_pred_unbatched_graded_mixed = ops.array(
            [0.9, 0.1, 0.7, 0.8], dtype="float32"
        )

        # === Batched Inputs ===
        self.y_true_batched = ops.array(
            [
                [0, 0, 1, 0],
                [1, 0, 3, 2],
                [0, 0, 0, 0],
                [2, 1, 0, 0],
            ],
            dtype="float32",
        )
        self.y_pred_batched = ops.array(
            [
                [0.1, 0.2, 0.9, 0.3],
                [0.1, 0.8, 0.9, 0.7],
                [0.4, 0.3, 0.2, 0.1],
                [0.9, 0.7, 0.1, 0.2],
            ],
            dtype="float32",
        )
        dcg_r0 = (math.pow(2, 1) - 1) / math.log2(1 + 1)
        dcg_r1 = (
            (math.pow(2, 3) - 1) / math.log2(1 + 1)
            + (math.pow(2, 2) - 1) / math.log2(3 + 1)
            + (math.pow(2, 1) - 1) / math.log2(4 + 1)
        )
        dcg_r2 = 0.0
        dcg_r3 = (math.pow(2, 2) - 1) / math.log2(1 + 1) + (
            math.pow(2, 1) - 1
        ) / math.log2(2 + 1)
        self.exp_value_batched = (dcg_r0 + dcg_r1 + dcg_r2 + dcg_r3) / 4.0

    def test_invalid_k_init(self):
        with self.assertRaisesRegex(
            ValueError, "`k` should be a positive integer"
        ):
            DCG(k=0)
        with self.assertRaisesRegex(
            ValueError, "`k` should be a positive integer"
        ):
            DCG(k=-5)
        with self.assertRaisesRegex(
            ValueError, "`k` should be a positive integer"
        ):
            DCG(k=3.5)

    def test_unbatched_perfect_rank_binary(self):
        dcg_metric = DCG()
        dcg_metric.update_state(
            self.y_true_unbatched, self.y_pred_unbatched_perfect
        )
        result = dcg_metric.result()
        exp_value = (math.pow(2, 1) - 1) / math.log2(1 + 1)
        self.assertAllClose(result, exp_value)

    def test_unbatched_good_rank_binary(self):
        dcg_metric = DCG()
        dcg_metric.update_state(
            self.y_true_unbatched, self.y_pred_unbatched_good
        )
        result = dcg_metric.result()
        exp_value = (math.pow(2, 1) - 1) / math.log2(2 + 1)
        self.assertAllClose(result, exp_value)

    def test_unbatched_bad_rank_binary(self):
        dcg_metric = DCG()
        dcg_metric.update_state(
            self.y_true_unbatched, self.y_pred_unbatched_bad
        )
        result = dcg_metric.result()
        exp_value = (math.pow(2, 1) - 1) / math.log2(3 + 1)
        self.assertAllClose(result, exp_value)

    def test_unbatched_no_relevant(self):
        dcg_metric = DCG()
        dcg_metric.update_state(
            self.y_true_unbatched_none,
            self.y_pred_unbatched_perfect,
        )
        result = dcg_metric.result()
        self.assertAllClose(result, 0.0)

    def test_unbatched_perfect_rank_graded(self):
        dcg_metric = DCG()
        dcg_metric.update_state(
            self.y_true_unbatched_graded,
            self.y_pred_unbatched_graded_perfect,
        )
        result = dcg_metric.result()
        exp_value = (
            (math.pow(2, 3) - 1) / math.log2(1 + 1)
            + (math.pow(2, 2) - 1) / math.log2(3 + 1)
            + (math.pow(2, 1) - 1) / math.log2(4 + 1)
        )
        self.assertAllClose(result, exp_value)

    def test_unbatched_mixed_rank_graded(self):
        dcg_metric = DCG()
        dcg_metric.update_state(
            self.y_true_unbatched_graded, self.y_pred_unbatched_graded_mixed
        )
        result = dcg_metric.result()
        exp_value = (
            (math.pow(2, 1) - 1) / math.log2(1 + 1)
            + (math.pow(2, 2) - 1) / math.log2(2 + 1)
            + (math.pow(2, 3) - 1) / math.log2(3 + 1)
        )
        self.assertAllClose(result, exp_value)

    def test_batched_input(self):
        dcg_metric = DCG()
        dcg_metric.update_state(self.y_true_batched, self.y_pred_batched)
        result = dcg_metric.result()
        self.assertAllClose(result, self.exp_value_batched)

    @parameterized.named_parameters(
        ("1", 1, 2.75),
        ("2", 2, 2.90773),
        ("3", 3, 3.28273),
        ("4", 4, 3.39040),
    )
    def test_k(self, k, exp_value_val):
        dcg_metric = DCG(k=k)
        dcg_metric.update_state(self.y_true_batched, self.y_pred_batched)
        result = dcg_metric.result()
        self.assertAllClose(result, exp_value_val, rtol=1e-5)

    def test_statefulness(self):
        dcg_metric = DCG()
        # Batch 1
        dcg_metric.update_state(
            self.y_true_batched[:2], self.y_pred_batched[:2]
        )
        result1 = dcg_metric.result()
        expected1_r0 = (math.pow(2, 1) - 1) / math.log2(1 + 1)
        expected1_r1 = (
            (math.pow(2, 3) - 1) / math.log2(1 + 1)
            + (math.pow(2, 2) - 1) / math.log2(3 + 1)
            + (math.pow(2, 1) - 1) / math.log2(4 + 1)
        )
        expected1 = (expected1_r0 + expected1_r1) / 2.0
        self.assertAllClose(result1, expected1)

        # Batch 2
        dcg_metric.update_state(
            self.y_true_batched[2:], self.y_pred_batched[2:]
        )
        result2 = dcg_metric.result()
        self.assertAllClose(result2, self.exp_value_batched)

        # Reset state
        dcg_metric.reset_state()
        result3 = dcg_metric.result()
        self.assertAllClose(result3, 0.0)

    @parameterized.named_parameters(
        ("0.5", 0.5, 3.390402),
        ("0", 0.0, 0.0),
        ("2", 2.0, 3.390402),
    )
    def test_scalar_sample_weight(self, sample_weight, expected_output):
        dcg_metric = DCG()
        dcg_metric.update_state(
            self.y_true_batched,
            self.y_pred_batched,
            sample_weight=sample_weight,
        )
        result = dcg_metric.result()
        self.assertAllClose(result, expected_output, rtol=1e-5)

    def test_1d_sample_weight(self):
        dcg_metric = DCG()
        sample_weight = ops.array([1.0, 0.5, 2.0, 1.5], dtype="float32")
        dcg_metric.update_state(
            self.y_true_batched,
            self.y_pred_batched,
            sample_weight=sample_weight,
        )
        result = dcg_metric.result()

        dcg_r0 = (math.pow(2, 1) - 1) / math.log2(1 + 1)
        dcg_r1 = (
            (math.pow(2, 3) - 1) / math.log2(1 + 1)
            + (math.pow(2, 2) - 1) / math.log2(3 + 1)
            + (math.pow(2, 1) - 1) / math.log2(4 + 1)
        )
        dcg_r2 = 0.0
        dcg_r3 = (math.pow(2, 2) - 1) / math.log2(1 + 1) + (
            math.pow(2, 1) - 1
        ) / math.log2(2 + 1)
        print(dcg_r0, dcg_r1, dcg_r2, dcg_r3)
        expected_val = (
            dcg_r0 * 1.0 + dcg_r1 * 0.5 + dcg_r2 * 1.0 + dcg_r3 * 1.5
        ) / (1.0 + 0.5 + 1.0 + 1.5)
        self.assertAllClose(result, expected_val, rtol=1e-5)

    @parameterized.named_parameters(
        (
            "mask_relevant_item",
            ops.array([[0, 1, 0]], dtype="float32"),
            ops.array([[0.5, 0.8, 0.2]], dtype="float32"),
            ops.array([[1.0, 0.0, 1.0]], dtype="float32"),
            0.0,
        ),
        (
            "mask_highest_ranked_item",
            ops.array([[0, 1, 0]], dtype="float32"),
            ops.array([[0.5, 0.8, 0.2]], dtype="float32"),
            ops.array([[1.0, 0.0, 1.0]], dtype="float32"),
            0.0,
        ),
        (
            "mask_lower_ranked_relevant",
            ops.array([[1, 0, 1]], dtype="float32"),
            ops.array([[0.8, 0.2, 0.6]], dtype="float32"),
            ops.array([[1.0, 1.0, 0.0]], dtype="float32"),
            (math.pow(2, 1) - 1) * 1.0 / math.log2(1 + 1),
        ),
        (
            "mask_irrelevant_item",
            ops.array([[0, 1, 0]], dtype="float32"),
            ops.array([[0.5, 0.8, 0.2]], dtype="float32"),
            ops.array([[0.0, 1.0, 1.0]], dtype="float32"),
            (math.pow(2, 1) - 1) * 1.0 / math.log2(1 + 1),
        ),
        (
            "varied_masks",
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
                    [0.5, 1.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0],
                ],
                dtype="float32",
            ),
            (
                ((math.pow(2, 1) - 1) * 1.0 / math.log2(2 + 1))
                + ((math.pow(2, 1) - 1) * 1.0 / math.log2(1 + 1))
            )
            / 2.0,
        ),
    )
    def test_2d_sample_weight_masking(
        self, y_true, y_pred, sample_weight, expected_output_formula
    ):
        dcg_metric = DCG()
        dcg_metric.update_state(y_true, y_pred, sample_weight=sample_weight)
        result = dcg_metric.result()
        self.assertAllClose(result, expected_output_formula, rtol=1e-5)

    def test_serialization(self):
        metric = DCG()
        restored = deserialize(serialize(metric))
        self.assertDictEqual(metric.get_config(), restored.get_config())

    def test_alternative_gain_rank_discount_fns(self):
        def linear_gain_fn(label):
            return ops.cast(label, dtype="float32")

        def inverse_pos_discount_fn(rank):
            return ops.divide(1.0, rank)

        dcg_metric = DCG(
            gain_fn=linear_gain_fn, rank_discount_fn=inverse_pos_discount_fn
        )
        dcg_metric.update_state(self.y_true_batched, self.y_pred_batched)
        result = dcg_metric.result()

        exp_value_r0 = 1.0 / 1.0
        exp_value_r1 = 3.0 / 1.0 + 2.0 / 3.0 + 1.0 / 4.0
        exp_value_r2 = 0.0
        exp_value_r3 = 2.0 / 1.0 + 1.0 / 2.0
        expected_avg_dcg = (
            exp_value_r0 + exp_value_r1 + exp_value_r2 + exp_value_r3
        ) / 4.0
        self.assertAllClose(result, expected_avg_dcg, rtol=1e-5)
