import keras
from absl.testing import parameterized
from keras import ops
from keras.losses import deserialize
from keras.losses import serialize

from keras_rs.src import testing
from keras_rs.src.losses.pairwise_soft_zero_one_loss import (
    PairwiseSoftZeroOneLoss,
)


class PairwiseSoftZeroOneLossTest(testing.TestCase, parameterized.TestCase):
    def setUp(self):
        self.unbatched_scores = ops.array([1.0, 3.0, 2.0, 4.0, 0.8])
        self.unbatched_labels = ops.array([1.0, 0.0, 1.0, 3.0, 2.0])

        self.batched_scores = ops.array(
            [[1.0, 3.0, 2.0, 4.0, 0.8], [1.0, 1.8, 2.0, 3.0, 2.0]]
        )
        self.batched_labels = ops.array(
            [[1.0, 0.0, 1.0, 3.0, 2.0], [0.0, 1.0, 2.0, 3.0, 1.5]]
        )

        self.expected_output = ops.array(
            [
                [0.880797, 0.0, 0.731059, 0.474736, 2.218608],
                [0.0, 0.310025, 1.219108, 0.888561, 0.719108],
            ]
        )

    def test_unbatched_input(self):
        loss = PairwiseSoftZeroOneLoss(reduction="none")
        output = loss(
            y_true=self.unbatched_labels, y_pred=self.unbatched_scores
        )
        self.assertAllClose(output, [self.expected_output[0]], atol=1e-5)

    def test_batched_input(self):
        loss = PairwiseSoftZeroOneLoss(reduction="none")
        output = loss(y_true=self.batched_labels, y_pred=self.batched_scores)
        self.assertAllClose(output, self.expected_output, atol=1e-5)

    def test_invalid_input_rank(self):
        rank_1_input = ops.ones((2, 3, 4))

        loss = PairwiseSoftZeroOneLoss()
        with self.assertRaises(ValueError):
            loss(y_true=rank_1_input, y_pred=rank_1_input)

    def test_loss_reduction(self):
        loss = PairwiseSoftZeroOneLoss(reduction="sum_over_batch_size")
        output = loss(y_true=self.batched_labels, y_pred=self.batched_scores)
        self.assertAlmostEqual(ops.convert_to_numpy(output), 0.744200, places=5)

    def test_scalar_sample_weight(self):
        sample_weight = ops.array(5.0)
        loss = PairwiseSoftZeroOneLoss(reduction="none")
        output = loss(
            y_true=self.batched_labels,
            y_pred=self.batched_scores,
            sample_weight=sample_weight,
        )
        self.assertAllClose(
            output, self.expected_output * sample_weight, atol=1e-5
        )

    def test_itemwise_sample_weight_with_zeros(self):
        sample_weight = ops.array(
            [[1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0, 0.0]]
        )
        loss = PairwiseSoftZeroOneLoss(reduction="none")
        output = loss(
            y_true=self.batched_labels,
            y_pred=self.batched_scores,
            sample_weight=sample_weight,
        )
        self.assertAllClose(
            output, self.expected_output * sample_weight, atol=1e-5
        )

    def test_mask_input(self):
        mask = ops.array(
            [[True, True, True, True, True], [True, True, True, False, False]]
        )
        loss = PairwiseSoftZeroOneLoss(reduction="none")
        output = loss(
            y_true={
                "labels": self.batched_labels,
                "mask": mask,
            },
            y_pred=self.batched_scores,
        )
        expected_output = ops.array(
            [
                [0.880797, 0.0, 0.731059, 0.474736, 2.218608],
                [0.0, 0.310025, 0.719108, 0.0, 0.0],
            ]
        )
        self.assertAllClose(output, expected_output, atol=1e-5)

    def test_model_fit(self):
        inputs = keras.Input(shape=(20,), dtype="float32")
        outputs = keras.layers.Dense(5)(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)

        model.compile(loss=PairwiseSoftZeroOneLoss(), optimizer="adam")
        model.fit(
            x=keras.random.normal((2, 20)),
            y=keras.random.randint((2, 5), minval=0, maxval=2),
        )

    def test_serialization(self):
        loss = PairwiseSoftZeroOneLoss()
        restored = deserialize(serialize(loss))
        self.assertDictEqual(loss.get_config(), restored.get_config())
