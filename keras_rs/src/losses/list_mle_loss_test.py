import keras
from absl.testing import parameterized
from keras import ops
from keras.losses import deserialize
from keras.losses import serialize

from keras_rs.src import testing
from keras_rs.src.losses.list_mle_loss import ListMLELoss


class ListMLELossTest(testing.TestCase, parameterized.TestCase):
    def setUp(self):
        self.unbatched_scores = ops.array([1.0, 3.0, 2.0, 4.0, 0.8])
        self.unbatched_labels = ops.array([1.0, 0.0, 1.0, 3.0, 2.0])

        self.batched_scores = ops.array(
            [[1.0, 3.0, 2.0, 4.0, 0.8], [1.0, 1.8, 2.0, 3.0, 2.0]]
        )
        self.batched_labels = ops.array(
            [[1.0, 0.0, 1.0, 3.0, 2.0], [0.0, 1.0, 2.0, 3.0, 1.5]]
        )
        self.expected_output = ops.array([6.865693, 3.088192])

    def test_unbatched_input(self):
        loss = ListMLELoss(reduction="none")
        output = loss(
            y_true=self.unbatched_labels, y_pred=self.unbatched_scores
        )
        self.assertEqual(output.shape, (1,))
        self.assertTrue(ops.convert_to_numpy(output[0]) > 0)
        self.assertAllClose(output, [self.expected_output[0]], atol=1e-5)

    def test_batched_input(self):
        loss = ListMLELoss(reduction="none")
        output = loss(y_true=self.batched_labels, y_pred=self.batched_scores)
        self.assertEqual(output.shape, (2,))
        self.assertTrue(ops.convert_to_numpy(output[0]) > 0)
        self.assertTrue(ops.convert_to_numpy(output[1]) > 0)
        self.assertAllClose(output, self.expected_output, atol=1e-5)

    def test_temperature(self):
        loss_temp = ListMLELoss(temperature=0.5, reduction="none")
        output_temp = loss_temp(
            y_true=self.batched_labels, y_pred=self.batched_scores
        )

        self.assertAllClose(
            output_temp,
            [10.969891, 2.1283305],
            atol=1e-5,
        )

    def test_invalid_input_rank(self):
        rank_1_input = ops.ones((2, 3, 4))

        loss = ListMLELoss()
        with self.assertRaises(ValueError):
            loss(y_true=rank_1_input, y_pred=rank_1_input)

    def test_loss_reduction(self):
        loss = ListMLELoss(reduction="sum_over_batch_size")
        output = loss(y_true=self.batched_labels, y_pred=self.batched_scores)

        self.assertAlmostEqual(
            ops.convert_to_numpy(output), 4.9769425, places=5
        )

    def test_scalar_sample_weight(self):
        sample_weight = ops.array(5.0)
        loss = ListMLELoss(reduction="none")

        output = loss(
            y_true=self.batched_labels,
            y_pred=self.batched_scores,
            sample_weight=sample_weight,
        )

        self.assertAllClose(
            output, self.expected_output * sample_weight, atol=1e-5
        )

    def test_model_fit(self):
        inputs = keras.Input(shape=(20,), dtype="float32")
        outputs = keras.layers.Dense(5)(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)

        model.compile(loss=ListMLELoss(), optimizer="adam")
        model.fit(
            x=keras.random.normal((2, 20)),
            y=keras.random.randint((2, 5), minval=0, maxval=2),
        )

    def test_serialization(self):
        loss = ListMLELoss(temperature=0.8)
        restored = deserialize(serialize(loss))
        self.assertDictEqual(loss.get_config(), restored.get_config())
