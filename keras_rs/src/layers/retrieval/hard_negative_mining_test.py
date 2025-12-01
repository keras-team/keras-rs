import keras
from absl.testing import parameterized
from keras import ops
from keras.layers import deserialize
from keras.layers import serialize

from keras_rs.src import testing
from keras_rs.src.layers.retrieval import hard_negative_mining


class HardNegativeMiningTest(testing.TestCase, parameterized.TestCase):
    def setUp(self):
        super().setUp()

    def create_inputs(self, rank=2):
        shape_3d = (15, 20, 10)
        shape = shape_3d[-rank:]

        rng = keras.random.SeedGenerator(42)
        logits = keras.random.uniform(shape, dtype="float32", seed=rng)
        num_candidates = shape[-1]
        labels = keras.random.randint(shape[0:-1], 0, num_candidates)
        labels = keras.ops.one_hot(labels, num_candidates, dtype="float32")

        return logits, labels

    @parameterized.named_parameters(
        [
            {
                "testcase_name": "rank_1_num_hard_negatives_3",
                "rank": 1,
                "num_hard_negatives": 3,
            },
            {
                "testcase_name": "rank_1_num_hard_negatives_30",
                "rank": 1,
                "num_hard_negatives": 30,
            },
            {
                "testcase_name": "rank_2_num_hard_negatives_3",
                "rank": 2,
                "num_hard_negatives": 3,
            },
            {
                "testcase_name": "rank_2_num_hard_negatives_30",
                "rank": 2,
                "num_hard_negatives": 30,
            },
            {
                "testcase_name": "rank_3_num_hard_negatives_3",
                "rank": 3,
                "num_hard_negatives": 3,
            },
            {
                "testcase_name": "rank_3_num_hard_negatives_30",
                "rank": 3,
                "num_hard_negatives": 30,
            },
        ]
    )
    def test_call(self, rank, num_hard_negatives):
        logits, labels = self.create_inputs(rank=rank)
        num_logits = logits.shape[-1]

        out_logits, out_labels = hard_negative_mining.HardNegativeMining(
            num_hard_negatives
        )(logits, labels)

        self.assertEqual(
            out_logits.shape[-1], min(num_hard_negatives + 1, num_logits)
        )

        # Logits for positives are always returned.
        self.assertAllClose(
            ops.sum(out_logits * out_labels, axis=-1),
            ops.sum(logits * labels, axis=-1),
        )

        # Set the logits for labels to be highest to ignore effect of labels.
        logits = logits + labels * 1000.0

        out_logits, _ = hard_negative_mining.HardNegativeMining(
            num_hard_negatives
        )(logits, labels)

        # Highest K logits are always returned.
        self.assertAllClose(
            ops.sort(logits, axis=-1)[..., -num_hard_negatives - 1 :],
            ops.sort(out_logits),
        )

    def test_predict(self):
        logits, labels = self.create_inputs()

        in_logits = keras.layers.Input(shape=logits.shape[1:])
        in_labels = keras.layers.Input(shape=labels.shape[1:])
        out_logits, out_labels = hard_negative_mining.HardNegativeMining(
            num_hard_negatives=3
        )(in_logits, in_labels)
        model = keras.Model([in_logits, in_labels], [out_logits, out_labels])

        model.predict([logits, labels], batch_size=8)

    def test_serialization(self):
        layer = hard_negative_mining.HardNegativeMining(num_hard_negatives=3)
        restored = deserialize(serialize(layer))
        self.assertDictEqual(layer.get_config(), restored.get_config())

    def test_model_saving(self):
        logits, labels = self.create_inputs()

        in_logits = keras.layers.Input(shape=logits.shape[1:])
        in_labels = keras.layers.Input(shape=labels.shape[1:])
        out_logits, out_labels = hard_negative_mining.HardNegativeMining(
            num_hard_negatives=3
        )(in_logits, in_labels)
        model = keras.Model([in_logits, in_labels], [out_logits, out_labels])

        self.run_model_saving_test(model=model, input_data=[logits, labels])
