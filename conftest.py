import pytest

from keras_rs.src.utils import tpu_test_utils


@pytest.fixture(scope="session", autouse=True)
def prime_shared_tpu_strategy(request):
    """
    Eagerly initializes the shared TPU strategy at the beginning of the session
    if running on a TPU. This helps catch initialization errors early.
    """
    strategy = tpu_test_utils.get_shared_tpu_strategy()
    if not strategy:
        pytest.fail(
            "Failed to initialize shared TPUStrategy for the test session. "
            "Check logs for details from create_tpu_strategy."
        )
