import keras
from absl.testing import parameterized
from keras import ops

from keras_rs.src import testing
from keras_rs.src.layers.loss import hard_negative_mining


class HardNegativeMiningTest(testing.TestCase, parameterized.TestCase):
    @parameterized.parameters(42, 123, 8391, 12390, 1230)
    def test_hard_negative_mining(self, random_seed):
        num_hard_negatives = 3
        # (num_queries, num_candidates)
        shape = (2, 20)
        rng = keras.random.SeedGenerator(random_seed)

        logits = keras.random.uniform(shape, dtype="float32", seed=rng)
        labels = ops.transpose(
            keras.random.shuffle(
                ops.transpose(ops.eye(*shape, dtype="float32")), seed=rng
            )
        )

        out_logits, out_labels = hard_negative_mining.HardNegativeMining(
            num_hard_negatives
        )(logits, labels)

        self.assertEqual(out_logits.shape[-1], num_hard_negatives + 1)

        # Logits for positives are always returned.
        self.assertAllClose(
            ops.sum(out_logits * out_labels, axis=1),
            ops.sum(logits * labels, axis=1),
        )

        # Set the logits for labels to be highest to ignore effect of labels.
        logits = logits + labels * 1000.0

        out_logits, _ = hard_negative_mining.HardNegativeMining(
            num_hard_negatives
        )(logits, labels)

        # Highest K logits are always returned.
        self.assertAllClose(
            ops.sort(logits, axis=1)[:, -num_hard_negatives - 1 :],
            ops.sort(out_logits),
        )
