import math

from absl.testing import parameterized
from keras import ops
from keras.metrics import deserialize
from keras.metrics import serialize

from keras_rs.src import testing
from keras_rs.src.metrics.n_dcg import nDCG


class nDCGTest(testing.TestCase, parameterized.TestCase):
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
        idcg_r0 = (math.pow(2, 1) - 1) / math.log2(1 + 1)
        ndcg_r0 = dcg_r0 / idcg_r0 if idcg_r0 > 0 else 0.0

        dcg_r1 = (
            (math.pow(2, 3) - 1) / math.log2(1 + 1)
            + (math.pow(2, 2) - 1) / math.log2(3 + 1)
            + (math.pow(2, 1) - 1) / math.log2(4 + 1)
        )
        idcg_r1 = (
            (math.pow(2, 3) - 1) / math.log2(1 + 1)
            + (math.pow(2, 2) - 1) / math.log2(2 + 1)
            + (math.pow(2, 1) - 1) / math.log2(3 + 1)
        )
        ndcg_r1 = dcg_r1 / idcg_r1 if idcg_r1 > 0 else 0.0

        ndcg_r2 = 0.0

        dcg_r3 = (math.pow(2, 2) - 1) / math.log2(1 + 1) + (
            math.pow(2, 1) - 1
        ) / math.log2(2 + 1)
        idcg_r3 = (math.pow(2, 2) - 1) / math.log2(1 + 1) + (
            math.pow(2, 1) - 1
        ) / math.log2(2 + 1)
        ndcg_r3 = dcg_r3 / idcg_r3 if idcg_r3 > 0 else 0.0

        self.exp_value_batched = (ndcg_r0 + ndcg_r1 + ndcg_r2 + ndcg_r3) / 4.0

    def test_invalid_k_init(self):
        with self.assertRaisesRegex(
            ValueError, "`k` should be a positive integer"
        ):
            nDCG(k=0)
        with self.assertRaisesRegex(
            ValueError, "`k` should be a positive integer"
        ):
            nDCG(k=-5)
        with self.assertRaisesRegex(
            ValueError, "`k` should be a positive integer"
        ):
            nDCG(k=3.5)

    def test_unbatched_perfect_rank_binary(self):
        ndcg_metric = nDCG()
        ndcg_metric.update_state(
            self.y_true_unbatched, self.y_pred_unbatched_perfect
        )
        result = ndcg_metric.result()
        self.assertAllClose(result, 1.0)

    def test_unbatched_good_rank_binary(self):
        ndcg_metric = nDCG()
        ndcg_metric.update_state(
            self.y_true_unbatched, self.y_pred_unbatched_good
        )
        result = ndcg_metric.result()
        ndcg = (math.pow(2, 1) - 1) / math.log2(2 + 1)
        idcg = (math.pow(2, 1) - 1) / math.log2(1 + 1)
        self.assertAllClose(result, ndcg / idcg if idcg > 0 else 0.0)

    def test_unbatched_bad_rank_binary(self):
        ndcg_metric = nDCG()
        ndcg_metric.update_state(
            self.y_true_unbatched, self.y_pred_unbatched_bad
        )
        result = ndcg_metric.result()
        ndcg = (math.pow(2, 1) - 1) / math.log2(3 + 1)
        idcg = (math.pow(2, 1) - 1) / math.log2(1 + 1)
        self.assertAllClose(result, ndcg / idcg if idcg > 0 else 0.0)

    def test_unbatched_no_relevant(self):
        ndcg_metric = nDCG()
        ndcg_metric.update_state(
            self.y_true_unbatched_none,
            self.y_pred_unbatched_perfect,
        )
        result = ndcg_metric.result()
        self.assertAllClose(result, 0.0)

    def test_unbatched_perfect_rank_graded(self):
        ndcg_metric = nDCG()
        ndcg_metric.update_state(
            self.y_true_unbatched_graded,
            self.y_pred_unbatched_graded_perfect,
        )
        result = ndcg_metric.result()
        dcg_val = (
            (math.pow(2, 3) - 1) / math.log2(1 + 1)
            + (math.pow(2, 2) - 1) / math.log2(3 + 1)
            + (math.pow(2, 1) - 1) / math.log2(4 + 1)
        )
        idcg_val = (
            (math.pow(2, 3) - 1) / math.log2(1 + 1)
            + (math.pow(2, 2) - 1) / math.log2(2 + 1)
            + (math.pow(2, 1) - 1) / math.log2(3 + 1)
        )
        exp_value = dcg_val / idcg_val if idcg_val > 0 else 0.0
        self.assertAllClose(result, exp_value, rtol=1e-5)

    def test_unbatched_mixed_rank_graded(self):
        ndcg_metric = nDCG()
        ndcg_metric.update_state(
            self.y_true_unbatched_graded, self.y_pred_unbatched_graded_mixed
        )
        result = ndcg_metric.result()
        dcg_val = (
            (math.pow(2, 1) - 1) / math.log2(1 + 1)
            + (math.pow(2, 2) - 1) / math.log2(2 + 1)
            + (math.pow(2, 3) - 1) / math.log2(3 + 1)
        )
        idcg_val = (
            (math.pow(2, 3) - 1) / math.log2(1 + 1)
            + (math.pow(2, 2) - 1) / math.log2(2 + 1)
            + (math.pow(2, 1) - 1) / math.log2(3 + 1)
        )
        exp_value = dcg_val / idcg_val if idcg_val > 0 else 0.0
        self.assertAllClose(result, exp_value, rtol=1e-5)

    def test_batched_input(self):
        ndcg_metric = nDCG()
        ndcg_metric.update_state(self.y_true_batched, self.y_pred_batched)
        result = ndcg_metric.result()
        self.assertAllClose(result, self.exp_value_batched, rtol=1e-5)

    @parameterized.named_parameters(
        ("1", 1, 0.75),
        ("2", 2, 0.69679),
        ("3", 3, 0.72624),
        ("4", 4, 0.73770),
    )
    def test_k(self, k, exp_value_val):
        ndcg_metric = nDCG(k=k)
        ndcg_metric.update_state(self.y_true_batched, self.y_pred_batched)
        result = ndcg_metric.result()
        self.assertAllClose(result, exp_value_val, rtol=1e-5)

    def test_statefulness(self):
        ndcg_metric = nDCG()
        # Batch 1
        ndcg_metric.update_state(
            self.y_true_batched[:2], self.y_pred_batched[:2]
        )
        result1 = ndcg_metric.result()
        # Calculate expected nDCG for first 2 rows
        dcg_r0 = (math.pow(2, 1) - 1) / math.log2(1 + 1)
        idcg_r0 = (math.pow(2, 1) - 1) / math.log2(1 + 1)
        ndcg_r0 = dcg_r0 / idcg_r0 if idcg_r0 > 0 else 0.0
        dcg_r1 = (
            (math.pow(2, 3) - 1) / math.log2(1 + 1)
            + (math.pow(2, 2) - 1) / math.log2(3 + 1)
            + (math.pow(2, 1) - 1) / math.log2(4 + 1)
        )
        idcg_r1 = (
            (math.pow(2, 3) - 1) / math.log2(1 + 1)
            + (math.pow(2, 2) - 1) / math.log2(2 + 1)
            + (math.pow(2, 1) - 1) / math.log2(3 + 1)
        )
        ndcg_r1 = dcg_r1 / idcg_r1 if idcg_r1 > 0 else 0.0
        expected1 = (ndcg_r0 + ndcg_r1) / 2.0
        self.assertAllClose(result1, expected1, rtol=1e-5)

        # Batch 2
        ndcg_metric.update_state(
            self.y_true_batched[2:], self.y_pred_batched[2:]
        )
        result2 = ndcg_metric.result()
        self.assertAllClose(result2, self.exp_value_batched, rtol=1e-5)

        # Reset state
        ndcg_metric.reset_state()
        result3 = ndcg_metric.result()
        self.assertAllClose(result3, 0.0)

    @parameterized.named_parameters(
        ("0.5", 0.5, 0.7377),
        ("0", 0.0, 0.0),
        ("2", 2.0, 0.7377),
    )
    def test_scalar_sample_weight(self, sample_weight, expected_output):
        ndcg_metric = nDCG()
        ndcg_metric.update_state(
            self.y_true_batched,
            self.y_pred_batched,
            sample_weight=sample_weight,
        )
        result = ndcg_metric.result()
        self.assertAllClose(result, expected_output, rtol=1e-5)

    def test_1d_sample_weight(self):
        ndcg_metric = nDCG()
        sample_weight = ops.array([1.0, 0.5, 2.0, 1.5], dtype="float32")
        ndcg_metric.update_state(
            self.y_true_batched,
            self.y_pred_batched,
            sample_weight=sample_weight,
        )
        result = ndcg_metric.result()
        self.assertAllClose(result, 0.74385, rtol=1e-5)

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
            1.0,
        ),
        (
            "mask_irrelevant_item",
            ops.array([[0, 1, 0]], dtype="float32"),
            ops.array([[0.5, 0.8, 0.2]], dtype="float32"),
            ops.array([[0.0, 1.0, 1.0]], dtype="float32"),
            1.0,
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
            0.815465,
        ),
    )
    def test_2d_sample_weight_masking(
        self, y_true, y_pred, sample_weight, expected_output_val
    ):
        ndcg_metric = nDCG()
        ndcg_metric.update_state(y_true, y_pred, sample_weight=sample_weight)
        result = ndcg_metric.result()
        self.assertAllClose(result, expected_output_val, rtol=1e-5)

    def test_serialization(self):
        metric = nDCG()
        restored = deserialize(serialize(metric))
        self.assertDictEqual(metric.get_config(), restored.get_config())

    def test_alternative_gain_rank_discount_fns(self):
        def linear_gain_fn(label):
            return ops.cast(label, dtype="float32")

        def inverse_discount_fn(rank):
            return ops.divide(1.0, rank)

        ndcg_metric = nDCG(
            gain_fn=linear_gain_fn, rank_discount_fn=inverse_discount_fn
        )
        ndcg_metric.update_state(self.y_true_batched, self.y_pred_batched)
        result = ndcg_metric.result()

        dcg_alt_r0 = 1.0 / 1.0
        idcg_alt_r0 = 1.0 / 1.0
        ndcg_alt_r0 = dcg_alt_r0 / idcg_alt_r0 if idcg_alt_r0 > 0 else 0.0

        dcg_alt_r1 = 3.0 / 1.0 + 2.0 / 3.0 + 1.0 / 4.0
        idcg_alt_r1 = 3.0 / 1.0 + 2.0 / 2.0 + 1.0 / 3.0
        ndcg_alt_r1 = dcg_alt_r1 / idcg_alt_r1 if idcg_alt_r1 > 0 else 0.0

        ndcg_alt_r2 = 0.0

        dcg_alt_r3 = 2.0 / 1.0 + 1.0 / 2.0
        idcg_alt_r3 = 2.0 / 1.0 + 1.0 / 2.0
        ndcg_alt_r3 = dcg_alt_r3 / idcg_alt_r3 if idcg_alt_r3 > 0 else 0.0

        expected_avg_ndcg = (
            ndcg_alt_r0 + ndcg_alt_r1 + ndcg_alt_r2 + ndcg_alt_r3
        ) / 4.0
        self.assertAllClose(result, expected_avg_ndcg, rtol=1e-5)
