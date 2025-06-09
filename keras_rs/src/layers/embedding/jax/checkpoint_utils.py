"""A Wrapper over orbax CheckpointManager for Keras3 Jax TPU Embeddings."""

from typing import Any

import keras
import orbax.checkpoint as ocp
from etils import epath


class JaxKeras3CheckpointManager(ocp.CheckpointManager):
    """A wrapper over orbax CheckpointManager for Keras3 Jax TPU Embeddings."""

    def __init__(
        self,
        model: keras.Model,
        checkpoint_dir: epath.PathLike,
        max_to_keep: int,
        steps_per_epoch: int = 1,
        **kwargs: Any,
    ):
        options = ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep, enable_async_checkpointing=False, **kwargs
        )
        self._model = model
        self._steps_per_epoch = steps_per_epoch
        self._checkpoint_dir = checkpoint_dir
        super().__init__(checkpoint_dir, options=options)

    def _get_state(self) -> tuple[dict[str, Any], Any | None]:
        """Gets the model state and metrics"""
        model_state = self._model.get_state_tree()
        state = {}
        metrics = None
        for k, v in model_state.items():
            if k == "metrics_variables":
                metrics = v
            else:
                state[k] = v
        return state, metrics

    def save_state(self, epoch: int) -> None:
        """Saves the model to the checkpoint directory.

        Args:
          epoch: The epoch number at which the state is saved.
        """
        state, metrics_value = self._get_state()
        self.save(
            epoch * self._steps_per_epoch,
            args=ocp.args.StandardSave(item=state),
            metrics=metrics_value,
        )

    def restore_state(self, step: int | None = None) -> None:
        """Restores the model from the checkpoint directory.

        Args:
          step: The step .number to restore the state from. Default=None
            restores the latest step.
        """
        if step is None:
            step = self.latest_step()
        # Restore the model state only, not metrics.
        state, _ = self._get_state()
        restored_state = self.restore(
            step, args=ocp.args.StandardRestore(item=state)
        )
        self._model.set_state_tree(restored_state)


class JaxKeras3CheckpointCallback(keras.callbacks.Callback):
    """A callback for checkpointing and restoring state using Orbax."""

    def __init__(
        self,
        model: keras.Model,
        checkpoint_dir: epath.PathLike,
        max_to_keep: int,
        steps_per_epoch: int = 1,
        **kwargs: Any,
    ):
        if keras.backend.backend() != "jax":
            raise ValueError(
                "`JaxKeras3CheckpointCallback` is only supported on a "
                "`jax` backend."
            )
        self._checkpoint_manager = JaxKeras3CheckpointManager(
            model, checkpoint_dir, max_to_keep, steps_per_epoch, **kwargs
        )

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        if not self.model.built or not self.model.optimizer.built:
            raise ValueError(
                "To use `JaxKeras3CheckpointCallback`, your model and "
                "optimizer must be built before you call `fit()`."
            )
        latest_epoch = self._checkpoint_manager.latest_step()
        if latest_epoch is not None:
            self._checkpoint_manager.restore_state(step=latest_epoch)

    def on_epoch_end(
        self, epoch: int, logs: dict[str, Any] | None = None
    ) -> None:
        self._checkpoint_manager.save_state(epoch)
