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

    Example:

    ```python
    # Create the layer.
    sampling_probability_correction = (
        keras_rs.layers.SamplingProbabilityCorrection()
    )

    # Correct the logits based on the provided candidate sampling probability.
    logits = sampling_probability_correction(logits, probabilities)
    ```
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
            logits: The logits tensor to correct, typically
                `[batch_size, num_candidates]` but can have more dimensions or
                be 1D as `[num_candidates]`.
            candidate_sampling_probability: The sampling probability with the
                same shape as `logits`.

        Returns:
            The corrected logits with the same shape as the input logits.
        """
        return logits - ops.log(
            ops.clip(candidate_sampling_probability, self.epsilon, 1.0)
        )

    def get_config(self) -> dict[str, Any]:
        config: dict[str, Any] = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config
