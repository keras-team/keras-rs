import unittest

import keras
import numpy as np

from keras_rs.src import types


class TestCase(unittest.TestCase):
    """TestCase class for all Keras Recommenders tests."""

    def setUp(self) -> None:
        super().setUp()
        keras.config.disable_traceback_filtering()
        keras.utils.clear_session()

    def assertAllClose(
        self,
        actual: types.Tensor,
        desired: types.Tensor,
        atol: float = 1e-6,
        rtol: float = 1e-6,
        msg: str = "",
    ) -> None:
        """Verify that two tensors are close in value element by element.

        Args:
          actual: Actual tensor, the first tensor to compare.
          desired: Expected tensor, the second tensor to compare.
          atol: Absolute tolerance.
          rtol: Relative tolerance.
          msg: Optional error message.
        """
        if not isinstance(actual, np.ndarray):
            actual = keras.ops.convert_to_numpy(actual)
        if not isinstance(desired, np.ndarray):
            desired = keras.ops.convert_to_numpy(desired)
        np.testing.assert_allclose(
            actual, desired, atol=atol, rtol=rtol, err_msg=msg
        )

    def assertAllEqual(
        self, actual: types.Tensor, desired: types.Tensor, msg: str = ""
    ) -> None:
        """Verify that two tensors are equal in value element by element.

        Args:
          actual: Actual tensor, the first tensor to compare.
          desired: Expected tensor, the second tensor to compare.
          msg: Optional error message.
        """
        if not isinstance(actual, np.ndarray):
            actual = keras.ops.convert_to_numpy(actual)
        if not isinstance(desired, np.ndarray):
            desired = keras.ops.convert_to_numpy(desired)
        np.testing.assert_array_equal(actual, desired, err_msg=msg)
