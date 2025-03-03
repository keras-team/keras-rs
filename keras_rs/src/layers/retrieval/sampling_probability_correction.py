from typing import Any

import keras
from keras import ops

from keras_rs.src import types
from keras_rs.src.api_export import keras_rs_export


@keras_rs_export("keras_rs.layers.SamplingProbabilityCorrection")
class SamplingProbabilityCorrection(keras.layers.Layer):
    """Sampling probability correction.

    Corrects the logits to reflect the sampling probability of negatives.

    Args:
        epsilon: float. Small float added to sampling probability to avoid
            taking the log of zero. Defaults to 1e-6.
        **kwargs: Args to pass to the base class.
    """

    def __init__(self, epsilon: float = 1e-6, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.built = True

    def call(
        self,
        logits: types.Tensor,
        candidate_sampling_probability: types.Tensor,
    ) -> types.Tensor:
        """Corrects input logits to account for candidate sampling probability.

        Args:
            logits: The logits to correct.
            candidate_sampling_probability: The sampling probability.

        Returns:
            The corrected logits.
        """
        return logits - ops.log(
            ops.clip(candidate_sampling_probability, self.epsilon, 1.0)
        )

    def get_config(self) -> dict[str, Any]:
        config: dict[str, Any] = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config
