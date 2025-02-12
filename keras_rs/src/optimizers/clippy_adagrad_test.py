import jax
import keras
import pytest
import tensorflow as tf
from keras import ops

from keras_rs.src import testing
from keras_rs.src.optimizers import clippy_adagrad


class ClippyAdagradTest(testing.TestCase):
    def test_scalar_clip(self):
        self.assertAllClose(
            (0.42, 0.21),
            clippy_adagrad.shrink_by_references(
                tensor=2.0,
                references=[4.0],
                relative_factors=[0.1],
                absolute_factor=0.02,
            ),
        )
        self.assertAllClose(
            (0.42, 0.21),
            clippy_adagrad.shrink_by_references(
                tensor=2.0,
                references=[-4.0],
                relative_factors=[0.1],
                absolute_factor=0.02,
            ),
        )
        self.assertAllClose(
            (-0.42, 0.21),
            clippy_adagrad.shrink_by_references(
                tensor=-2.0,
                references=[4.0],
                relative_factors=[0.1],
                absolute_factor=0.02,
            ),
        )
        self.assertAllClose(
            (-0.4, 0.2),
            clippy_adagrad.shrink_by_references(
                tensor=-2.0,
                references=[4.0],
                relative_factors=[0.1],
                absolute_factor=0.0,
            ),
        )
        self.assertAllClose(
            (0.0, 0.0),
            clippy_adagrad.shrink_by_references(
                tensor=-2.0,
                references=[0.0],
                relative_factors=[0.1],
                absolute_factor=0.0,
            ),
        )
        # No clipping needed.
        self.assertAllClose(
            (2.0, 1.0),
            clippy_adagrad.shrink_by_references(
                tensor=2.0,
                references=[20.0],
                relative_factors=[0.1],
                absolute_factor=0.1,
            ),
        )
        self.assertAllClose(
            (-2.0, 1.0),
            clippy_adagrad.shrink_by_references(
                tensor=-2.0,
                references=[20.0],
                relative_factors=[0.1],
                absolute_factor=0.1,
            ),
        )
        self.assertAllClose(
            (0.0, 1.0),
            clippy_adagrad.shrink_by_references(
                tensor=0.0,
                references=[1.0],
                relative_factors=[0.1],
                absolute_factor=0.1,
            ),
        )
        self.assertAllClose(
            (0.0, 1.0),
            clippy_adagrad.shrink_by_references(
                tensor=0.0,
                references=[1.0],
                relative_factors=[0.1],
                absolute_factor=0.0,
            ),
        )
        self.assertAllClose(
            (0.0, 1.0),
            clippy_adagrad.shrink_by_references(
                tensor=0.0,
                references=[0.0],
                relative_factors=[0.0],
                absolute_factor=0.0,
            ),
        )

    def test_scalar_multiple_clip(self):
        self.assertAllClose(
            (4 * 0.1 + 5 * 0.2 + 0.02, (4 * 0.1 + 5 * 0.2 + 0.02) / 2),
            clippy_adagrad.shrink_by_references(
                tensor=2.0,
                references=[4.0, -5.0],
                relative_factors=[0.1, 0.2],
                absolute_factor=0.02,
            ),
        )

    def test_scalar_empty_reference(self):
        self.assertAllClose(
            (0.02, 0.01),
            clippy_adagrad.shrink_by_references(
                tensor=2.0,
                references=[],
                relative_factors=[],
                absolute_factor=0.02,
            ),
        )
        self.assertAllClose(
            (0.0, 1.0),
            clippy_adagrad.shrink_by_references(
                tensor=0.0,
                references=[],
                relative_factors=[],
                absolute_factor=0.0,
            ),
        )

    def test_tensor_clip(self):
        clipped, scale = clippy_adagrad.shrink_by_references(
            tensor=ops.array([1.0, 1.0]),
            references=[ops.array([1.0, 0.1])],
            relative_factors=[0.1],
            absolute_factor=0.01,
        )
        self.assertAllClose(ops.array([0.02, 0.02]), clipped)
        self.assertAllClose(0.02, scale)

    def test_tensor_clip_zero_absolute_factor(self):
        clipped, scale = clippy_adagrad.shrink_by_references(
            tensor=ops.array([1.0, 1.0, 0.0, 0.0]),
            references=[ops.array([1.0, 0.1, 1.0, 0.0])],
            relative_factors=[0.1],
            absolute_factor=0.0,
        )
        self.assertAllClose(ops.array([0.01, 0.01, 0.0, 0.0]), clipped)
        self.assertAllClose(0.01, scale)

    def test_tensor_clip_zero_reference(self):
        clipped, scale = clippy_adagrad.shrink_by_references(
            tensor=ops.array([1.0, 1.0, 0.0, 0.0]),
            references=[ops.array([1.0, 0.0, 1.0, 0.0])],
            relative_factors=[0.1],
            absolute_factor=0.0,
        )
        self.assertAllClose(ops.array([0.0, 0.0, 0.0, 0.0]), clipped)
        self.assertAllClose(0.0, scale)

    def test_broadcast(self):
        clipped, scale = clippy_adagrad.shrink_by_references(
            tensor=ops.array([[1.0, 2.0], [1.0, 2.0]]),
            references=[ops.array(1.0)],
            relative_factors=[0.1],
            absolute_factor=0.1,
        )
        self.assertAllClose(ops.array([[0.1, 0.2], [0.1, 0.2]]), clipped)
        self.assertAllClose(0.1, scale)

    def test_single_step_no_clip(self):
        learning_rate = 0.1
        initial_accumulator_sqrt = 0.1
        optimizer = clippy_adagrad.ClippyAdagrad(
            learning_rate=learning_rate,
            initial_accumulator_value=initial_accumulator_sqrt**2,
            export_clipping_factors=True,
        )
        x = keras.Variable([1.0, 2.0], dtype="float32")
        g = ops.array([0.1, 0.15])
        sparse_x = keras.Variable([[3.0, 4.0], [1.0, 2.0]], dtype="float32")
        sparse_g = ops.array([[0.0, 0.0], [0.1, 0.15]])
        optimizer.apply_gradients([(g, x), (sparse_g, sparse_x)])
        self.assertAllClose(
            x,
            ops.array(
                [
                    1.0 - learning_rate * 0.1 / initial_accumulator_sqrt,
                    2.0 - learning_rate * 0.15 / initial_accumulator_sqrt,
                ]
            ),
            atol=1e-6,
        )
        self.assertAllClose(
            sparse_x,
            ops.array(
                [
                    [3.0, 4.0],
                    [
                        1.0 - learning_rate * 0.1 / initial_accumulator_sqrt,
                        2.0 - learning_rate * 0.15 / initial_accumulator_sqrt,
                    ],
                ]
            ),
            atol=1e-6,
        )
        self.assertAllClose(
            optimizer._accumulators[0],
            ops.array(
                [
                    initial_accumulator_sqrt**2 + 0.1**2,
                    initial_accumulator_sqrt**2 + 0.15**2,
                ]
            ),
            atol=1e-6,
        )
        self.assertAllClose(
            optimizer._accumulators[1],
            ops.array(
                [
                    [initial_accumulator_sqrt**2, initial_accumulator_sqrt**2],
                    [
                        initial_accumulator_sqrt**2 + 0.1**2,
                        initial_accumulator_sqrt**2 + 0.15**2,
                    ],
                ]
            ),
            atol=1e-6,
        )
        for clipping_factor in optimizer.clipping_factors:
            self.assertAllClose(clipping_factor.value, 1.0, atol=1e-6)

    def test_single_step_clip(self):
        learning_rate = 0.2
        initial_accumulator_sqrt = 0.1
        optimizer = clippy_adagrad.ClippyAdagrad(
            learning_rate=learning_rate,
            initial_accumulator_value=initial_accumulator_sqrt**2,
            variable_relative_threshold=0.4,
            accumulator_relative_threshold=0.01,
            absolute_threshold=0.1,
            epsilon=0.0,
            export_clipping_factors=True,
        )
        x = keras.Variable([1.0, 2.0], dtype="float32")
        g = ops.array([10.0, 10.0], dtype="float32")
        sparse_x = keras.Variable([[3.0, 4.0], [1.0, 2.0]], dtype="float32")
        sparse_g = ops.array([[0.0, 0.0], [10.0, 10.0]])
        optimizer.apply_gradients([(g, x), (sparse_g, sparse_x)])
        # Gradient is clipped so change in x in each coordinate is at
        # most 0.4 x + 0.01 / initial_accumulator_sqrt + 0.1.
        self.assertAllClose(x, ops.array([0.4, 1.4]), atol=1e-6)
        self.assertAllClose(
            sparse_x,
            ops.array([[3.0, 4.0], [0.4, 1.4]]),
            atol=1e-6,
        )
        self.assertAllClose(
            optimizer._accumulators[0],
            ops.array(
                [
                    initial_accumulator_sqrt**2 + 10.0**2,
                    initial_accumulator_sqrt**2 + 10.0**2,
                ]
            ),
            atol=1e-6,
        )
        self.assertAllClose(
            optimizer._accumulators[1],
            ops.array(
                [
                    [initial_accumulator_sqrt**2, initial_accumulator_sqrt**2],
                    [
                        initial_accumulator_sqrt**2 + 10.0**2,
                        initial_accumulator_sqrt**2 + 10.0**2,
                    ],
                ]
            ),
            atol=1e-6,
        )
        # g * clipping_factor * learning_rate / initial_accumulator_sqrt == 0.6
        for clipping_factor in optimizer.clipping_factors:
            self.assertAllClose(
                clipping_factor.value,
                0.6 * initial_accumulator_sqrt / (10.0 * learning_rate),
                atol=1e-6,
            )

    def test_single_step_clip_with_accumulator(self):
        """Test clip_accumulator_update=True."""
        learning_rate = 0.2
        initial_accumulator_sqrt = 0.1
        optimizer = clippy_adagrad.ClippyAdagrad(
            learning_rate=learning_rate,
            initial_accumulator_value=initial_accumulator_sqrt**2,
            variable_relative_threshold=0.4,
            accumulator_relative_threshold=0.01,
            absolute_threshold=0.1,
            epsilon=0.0,
            export_clipping_factors=True,
            clip_accumulator_update=True,
        )
        x = keras.Variable([1.0, 2.0], dtype="float32")
        g = ops.array([10.0, 10.0], dtype="float32")
        sparse_x = keras.Variable([[3.0, 4.0], [1.0, 2.0]], dtype="float32")
        sparse_g = ops.array([[0.0, 0.0], [10.0, 10.0]])
        optimizer.apply_gradients([(g, x), (sparse_g, sparse_x)])
        # Gradient is clipped so change in x in each coordinate is at
        # most 0.4 x + 0.01 / initial_accumulator_sqrt + 0.1.
        self.assertAllClose(x, ops.array([0.4, 1.4]), atol=1e-6)
        self.assertAllClose(
            sparse_x,
            ops.array([[3.0, 4.0], [0.4, 1.4]]),
            atol=1e-6,
        )
        # Make sure the accumulator update takes the clipping factor into
        # account.
        self.assertAllClose(
            optimizer._accumulators[0],
            ops.array(
                [
                    initial_accumulator_sqrt**2
                    + (optimizer.clipping_factors[0] * 10) ** 2,
                    initial_accumulator_sqrt**2
                    + (optimizer.clipping_factors[0] * 10) ** 2,
                ]
            ),
            atol=1e-6,
        )
        self.assertAllClose(
            optimizer._accumulators[1],
            ops.array(
                [
                    [initial_accumulator_sqrt**2, initial_accumulator_sqrt**2],
                    [
                        initial_accumulator_sqrt**2
                        + (optimizer.clipping_factors[1] * 10) ** 2,
                        initial_accumulator_sqrt**2
                        + (optimizer.clipping_factors[1] * 10) ** 2,
                    ],
                ]
            ),
            atol=1e-6,
        )
        # g * clipping_factor * learning_rate / initial_accumulator_sqrt == 0.6
        for clipping_factor in optimizer.clipping_factors:
            self.assertAllClose(
                clipping_factor.value,
                0.6 * initial_accumulator_sqrt / (10.0 * learning_rate),
                atol=1e-6,
            )

    def test_single_step_clip_with_standard_update(self):
        """Test use_standard_accumulator_update=True."""
        learning_rate = 0.1
        initial_accumulator_sqrt = 0.0
        optimizer = clippy_adagrad.ClippyAdagrad(
            learning_rate=learning_rate,
            initial_accumulator_value=initial_accumulator_sqrt**2,
            export_clipping_factors=True,
            use_standard_accumulator_update=True,
        )
        x = keras.Variable([1.0, 2.0], dtype="float32")
        g = ops.array([0.1, 0.15])
        sparse_x = keras.Variable([[3.0, 4.0], [1.0, 2.0]], dtype="float32")
        sparse_g = ops.array([[0.0, 0.0], [0.1, 0.15]])
        optimizer.apply_gradients([(g, x), (sparse_g, sparse_x)])
        # Since the accumulator was initialized as zero, the Adagrad delta is 1.
        self.assertAllClose(
            x,
            ops.array([1.0 - learning_rate, 2.0 - learning_rate]),
            atol=1e-6,
        )
        self.assertAllClose(
            sparse_x,
            ops.array([[3.0, 4.0], [1.0 - learning_rate, 2.0 - learning_rate]]),
            atol=1e-6,
        )
        self.assertAllClose(optimizer._accumulators[0], ops.square(g))
        self.assertAllClose(
            optimizer._accumulators[1],
            ops.array(
                [
                    [initial_accumulator_sqrt**2, initial_accumulator_sqrt**2],
                    [0.1**2, 0.15**2],
                ]
            ),
            atol=1e-6,
        )
        for clipping_factor in optimizer.clipping_factors:
            self.assertAllClose(clipping_factor.value, 1.0, atol=1e-6)

    @pytest.mark.skipif(
        keras.backend.backend() == "torch", reason="No jit support on torch"
    )
    def test_jit(self):
        learning_rate = 0.1
        initial_accumulator_sqrt = 0.1
        optimizer = clippy_adagrad.ClippyAdagrad(
            learning_rate=learning_rate,
            initial_accumulator_value=initial_accumulator_sqrt**2,
        )

        x = keras.Variable([1.0, 2.0], dtype="float32", name="x")
        g = ops.array([0.1, 0.15])
        sparse_x = keras.Variable(
            [[3.0, 4.0], [1.0, 2.0]], dtype="float32", name="sparse_x"
        )
        sparse_g = ops.array([[0.0, 0.0], [0.1, 0.15]])
        optimizer.build((x, sparse_x))

        if keras.backend.backend() == "tensorflow":

            @tf.function(jit_compile=True)
            def _train_step():
                optimizer.apply_gradients([(g, x), (sparse_g, sparse_x)])

            _train_step()
            x_after = x.value
            sparse_x_after = sparse_x.value
            accumulator_0_after = optimizer._accumulators[0].value
            accumulator_1_after = optimizer._accumulators[1].value
        elif keras.backend.backend() == "jax":

            @jax.jit
            def _train_step(optimizer_variables, grads, trainable_variables):
                return optimizer.stateless_apply(
                    optimizer_variables, grads, trainable_variables
                )

            (x_after, sparse_x_after), optimizer_vars = _train_step(
                [v.value for v in optimizer.variables],
                (g, sparse_g),
                (x.value, sparse_x.value),
            )

            accumulator_0_after = None
            accumulator_1_after = None
            for var, value_after in zip(optimizer.variables, optimizer_vars):
                if var is optimizer._accumulators[0]:
                    accumulator_0_after = value_after
                elif var is optimizer._accumulators[1]:
                    accumulator_1_after = value_after
        else:
            raise ValueError(f"Unsupported backend: {keras.backend.backend()}")

        self.assertAllClose(
            x_after,
            ops.array(
                [
                    1.0 - learning_rate * 0.1 / initial_accumulator_sqrt,
                    2.0 - learning_rate * 0.15 / initial_accumulator_sqrt,
                ]
            ),
            atol=1e-6,
        )
        self.assertAllClose(
            sparse_x_after,
            ops.array(
                [
                    [3.0, 4.0],
                    [
                        1.0 - learning_rate * 0.1 / initial_accumulator_sqrt,
                        2.0 - learning_rate * 0.15 / initial_accumulator_sqrt,
                    ],
                ]
            ),
            atol=1e-6,
        )
        self.assertAllClose(
            accumulator_0_after,
            ops.array(
                [
                    initial_accumulator_sqrt**2 + 0.1**2,
                    initial_accumulator_sqrt**2 + 0.15**2,
                ]
            ),
            atol=1e-6,
        )
        self.assertAllClose(
            accumulator_1_after,
            ops.array(
                [
                    [initial_accumulator_sqrt**2, initial_accumulator_sqrt**2],
                    [
                        initial_accumulator_sqrt**2 + 0.1**2,
                        initial_accumulator_sqrt**2 + 0.15**2,
                    ],
                ]
            ),
            atol=1e-6,
        )

    def test_get_config(self):
        optimizer = clippy_adagrad.ClippyAdagrad(
            learning_rate=0.1,
            initial_accumulator_value=0.2,
            variable_relative_threshold=0.3,
            accumulator_relative_threshold=0.6,
            absolute_threshold=0.4,
            epsilon=0.5,
            export_clipping_factors=True,
        )
        config = optimizer.get_config()
        restored_optimizer = clippy_adagrad.ClippyAdagrad.from_config(config)
        self.assertEqual(
            optimizer.learning_rate, restored_optimizer.learning_rate
        )
        self.assertEqual(
            optimizer.initial_accumulator_value,
            restored_optimizer.initial_accumulator_value,
        )
        self.assertEqual(
            optimizer.variable_relative_threshold,
            restored_optimizer.variable_relative_threshold,
        )
        self.assertEqual(
            optimizer.absolute_threshold, restored_optimizer.absolute_threshold
        )
        self.assertEqual(optimizer.epsilon, restored_optimizer.epsilon)
        self.assertEqual(
            optimizer.export_clipping_factors,
            restored_optimizer.export_clipping_factors,
        )
        self.assertEqual(
            optimizer.accumulator_relative_threshold,
            restored_optimizer.accumulator_relative_threshold,
        )
