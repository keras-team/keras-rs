import argparse
import importlib
import logging
import os

os.environ["KERAS_BACKEND"] = "jax"

import keras

import keras_rs

from .dataloader import DataLoader
from .model import DLRMDCNV2

# Set random seed.
SEED = 1337

logger = logging.getLogger(__name__)

keras.utils.set_random_seed(SEED)
keras.config.disable_traceback_filtering()


class MetricLogger(keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        print("--->", logs["loss"])


def main(
    ds_cfg,
    model_cfg,
    training_cfg,
):
    passed_args = locals()
    logger.debug("Called `main()` with: %s", passed_args)

    # Set DDP as Keras distribution strategy
    devices = keras.distribution.list_devices(device_type="tpu")
    distribution = keras.distribution.DataParallel(devices=devices)
    keras.distribution.set_distribution(distribution)
    num_processes = distribution._num_process
    logger.info("Initialized distribution strategy.")
    logger.info("Found %d devices.", len(devices))
    logger.info("Running with %d processes.", num_processes)
    if distribution._process_id is not None:
        logger.info("Current Process ID: %d", distribution._process_id)

    # === Distributed embeddings' configs for lookup features ===

    # For features which have vocabulary_size < embedding_threshold, we can
    # just do a normal dense lookup for those instead of having distributed
    # embeddings. We could ideally pass `placement = default_device` to
    # `keras_rs.layers.TableConfig` directly (and wouldn't have to do this
    # separation of features), but doing it that way will necessarily require
    # a separate optimiser for the embedding layer.
    small_emb_features = []
    large_emb_features = []
    for emb_feature in ds_cfg.lookup:
        if emb_feature["vocabulary_size"] < model_cfg.embedding_threshold:
            small_emb_features.append(emb_feature)
        else:
            large_emb_features.append(emb_feature)
    logger.debug("Large Embedding Features: %s", large_emb_features)
    logger.debug("Small Embedding Features: %s", small_emb_features)

    feature_configs = {}
    for large_emb_feature in large_emb_features:
        feature_name = large_emb_feature["new_name"]
        vocabulary_size = large_emb_feature["vocabulary_size"]
        feature_list_length = large_emb_feature["feature_list_length"]

        table_config = keras_rs.layers.TableConfig(
            name=f"{feature_name}_table",
            vocabulary_size=vocabulary_size,
            embedding_dim=model_cfg.embedding_dim,
            # TODO(abheesht): Verify.
            initializer=keras.initializers.VarianceScaling(
                scale=1.0,
                mode="fan_in",
                distribution="uniform",
                seed=SEED,
            ),
            optimizer=keras.optimizers.Adagrad(
                learning_rate=model_cfg.learning_rate
            ),
            combiner="sum",
            placement="sparsecore",
            # TODO: These two args are not getting passed down to
            # `jax-tpu-embedding` properly, seems like.
            max_ids_per_partition=model_cfg.max_ids_per_partition,
            max_unique_ids_per_partition=model_cfg.max_unique_ids_per_partition,
        )
        feature_configs[f"{feature_name}_id"] = keras_rs.layers.FeatureConfig(
            name=feature_name,
            table=table_config,
            # TODO: Verify whether it should be `(bsz, 1)` or
            # `(bsz, feature_list_length)`. The original example uses 1.
            input_shape=(training_cfg.global_batch_size, 1),
            output_shape=(
                training_cfg.global_batch_size,
                model_cfg.embedding_dim,
            ),
        )

    # === Instantiate model ===
    # We instantiate the model first, because we need to preprocess large
    # embedding feature inputs using the distributed embedding layer defined
    # inside the model class.
    logger.info("Initialising model...")
    model = DLRMDCNV2(
        large_emb_feature_configs=feature_configs,
        small_emb_features=small_emb_features,
        embedding_dim=model_cfg.embedding_dim,
        bottom_mlp_dims=model_cfg.bottom_mlp_dims,
        top_mlp_dims=model_cfg.top_mlp_dims,
        num_dcn_layers=model_cfg.num_dcn_layers,
        dcn_projection_dim=model_cfg.dcn_projection_dim,
        seed=SEED,
        dtype="float32",
        name="dlrm_dcn_v2",
    )
    model.compile(
        loss=keras.losses.BinaryCrossentropy(),
        optimizer=keras.optimizers.Adagrad(
            learning_rate=training_cfg.learning_rate
        ),
        metrics=[keras.metrics.BinaryAccuracy()],
    )
    logger.info("Initialised model: %s", model)

    # === Load dataset ===
    logger.info("Loading dataset...")

    # Keras does not have a straightforward way to log at a step-level instead
    # of epoch-level. So, we do a workaround here.
    if ds_cfg.val_file_pattern:
        steps_per_epoch = training_cfg.eval_freq
        epochs = training_cfg.num_steps // training_cfg.eval_freq
        do_eval = True
    else:
        steps_per_epoch = training_cfg.num_steps
        epochs = 1
        do_eval = False

    train_ds = DataLoader(
        file_pattern=ds_cfg.file_pattern,
        batch_size=training_cfg.global_batch_size,
        file_batch_size=ds_cfg.get("file_batch_size", None),
        dense_features=ds_cfg.dense,
        large_emb_features=large_emb_features,
        small_emb_features=small_emb_features,
        label=ds_cfg.label,
        num_steps=steps_per_epoch + 20,
        training=True,
    ).create_dataset(
        process_id=distribution._process_id,
        num_processes=num_processes,
        shuffle_buffer=ds_cfg.get("shuffle_buffer", None),
    )
    if do_eval:
        eval_ds = DataLoader(
            file_pattern=ds_cfg.val_file_pattern,
            batch_size=training_cfg.global_batch_size,
            file_batch_size=ds_cfg.get("file_batch_size", None),
            dense_features=ds_cfg.dense,
            large_emb_features=large_emb_features,
            small_emb_features=small_emb_features,
            label=ds_cfg.label,
            num_steps=training_cfg.num_eval_steps,
            repeat=True,
            training=False,
        ).create_dataset(
            process_id=distribution._process_id,
            num_processes=num_processes,
        )
    # For the multi-host case, the dataset has to be distributed manually.
    # See note here:
    # https://github.com/keras-team/keras-rs/blob/main/keras_rs/src/layers/embedding/base_distributed_embedding.py#L352-L363.
    if num_processes > 1:
        train_ds = distribution.distribute_dataset(train_ds)
        if do_eval:
            eval_ds = distribution.distribute_dataset(eval_ds)
        distribution.auto_shard_dataset = False

    def generator(dataset, training=False):
        """Converts tf.data Dataset to a Python generator and preprocesses
        large embedding features.
        """
        for features, labels in dataset:
            preprocessed_large_embeddings = model.embedding_layer.preprocess(
                features["large_emb_inputs"], training=training
            )

            x = {
                "dense_input": features["dense_input"],
                "large_emb_inputs": preprocessed_large_embeddings,
                "small_emb_inputs": features["small_emb_inputs"],
            }
            y = labels
            yield (x, y)

    logger.info("Preprocessing large embedding tables...")
    train_generator = generator(train_ds, training=True)
    if do_eval:
        eval_generator = generator(eval_ds, training=False)
    logger.debug("Inspecting one batch of data...")
    for first_batch in train_generator:
        logger.debug("Dense inputs:%s", first_batch[0]["dense_input"])
        logger.debug(
            "Small embedding inputs:%s",
            first_batch[0]["small_emb_inputs"]["cat_39_id"],
        )
        logger.debug(
            "Large embedding inputs:%s", first_batch[0]["large_emb_inputs"]
        )
        break
    logger.info("Successfully preprocessed one batch of data")

    # === Training ===
    logger.info("Training...")
    model.fit(
        train_generator,
        # validation_data=eval_generator,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        # callbacks=[MetricLogger()],
        # validation_steps=training_cfg.num_eval_steps,
        # validation_freq=1,
        # verbose=0,
    )
    logger.info("Training finished")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("Launching train script...")
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the DLRM-DCNv2 model on the Criteo dataset (MLPerf)"
        )
    )
    parser.add_argument(
        "--config_name", type=str, help="Name of the `.py` config file."
    )
    args = parser.parse_args()

    logger.info("Reading config from %s", args.config_name)
    config = importlib.import_module(
        f".configs.{args.config_name}", package=__package__
    ).config
    logger.info("Config: %s", config)

    ds_cfg = config["dataset"]
    model_cfg = config["model"]
    training_cfg = config["training"]

    main(
        ds_cfg=ds_cfg,
        model_cfg=model_cfg,
        training_cfg=training_cfg,
    )

    logger.info("Train script finished")
