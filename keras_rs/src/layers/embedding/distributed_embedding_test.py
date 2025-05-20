import contextlib
import functools
import math
import os
import tempfile

import jax
import jax.experimental.sparse as jax_sparse
import jax.numpy as jnp
import keras
import numpy as np
import tensorflow as tf
from absl import flags
from absl.testing import parameterized

from keras_rs.src import testing
from keras_rs.src.layers import embedding
from keras_rs.src.layers.embedding import distributed_embedding_config as config

FLAGS = flags.FLAGS
_TPU = flags.DEFINE_string("tpu", None, "The TPU to use for TPUStrategy.")


FEATURE1_EMBEDDING_OUTPUT_DIM = 7
FEATURE2_EMBEDDING_OUTPUT_DIM = 11
EMBEDDING_OUTPUT_DIM = 7
VOCABULARY_SIZE = 23
BATCH_SIZE_PER_CORE = 16
SEQUENCE_LENGTH = 13


class DummyStrategy:
    def scope(self):
        return contextlib.nullcontext()

    @property
    def num_replicas_in_sync(self):
        return 1

    def run(self, fn, args):
        return fn(*args)

    def experimental_distribute_dataset(self, dataset, options=None):
        del options
        return dataset


class DistributedEmbeddingTest(testing.TestCase, parameterized.TestCase):
    def setUp(self):
        super().setUp()
        try:
            self.on_tpu = _TPU.value is not None
        except flags.UnparsedFlagAccessError:
            self.on_tpu = False

        self.placement = "sparsecore" if self.on_tpu else "default_device"

        if keras.backend.backend() == "tensorflow":
            tf.debugging.disable_traceback_filtering()

        if keras.backend.backend() == "tensorflow" and self.on_tpu:
            # FLAGS.xla_sparse_core_max_ids_per_partition_per_sample = 16
            # FLAGS.xla_sparse_core_max_unique_ids_per_partition_per_sample = 16

            resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
                tpu=_TPU.value
            )
            tf.config.experimental_connect_to_cluster(resolver)

            topology = tf.tpu.experimental.initialize_tpu_system(resolver)
            tpu_metadata = resolver.get_tpu_system_metadata()

            device_assignment = tf.tpu.experimental.DeviceAssignment.build(
                topology, num_replicas=tpu_metadata.num_hosts
            )
            self._strategy = tf.distribute.TPUStrategy(
                resolver, experimental_device_assignment=device_assignment
            )
            print("### num_replicas", self._strategy.num_replicas_in_sync)
            self.addCleanup(tf.tpu.experimental.shutdown_tpu_system, resolver)
        else:
            self._strategy = DummyStrategy()

    def run_with_strategy(self, fn, *args, jit_compile=False):
        """Wrapper for running a function under a strategy."""

        if keras.backend.backend() == "tensorflow":

            @tf.function(jit_compile=jit_compile)
            def tf_function_wrapper(*tf_function_args):
                def strategy_fn(*strategy_fn_args):
                    return fn(*strategy_fn_args)

                return self._strategy.run(strategy_fn, args=tf_function_args)

            return tf_function_wrapper(*args)
        else:
            self.assertFalse(jit_compile)
            return fn(*args)

    def get_embedding_config(self, input_type, placement):
        sequence_length = 1 if input_type == "dense" else SEQUENCE_LENGTH

        feature1_table = config.TableConfig(
            name="feature1_table",
            vocabulary_size=VOCABULARY_SIZE,
            embedding_dim=FEATURE1_EMBEDDING_OUTPUT_DIM,
            placement=placement,
        )
        feature2_table = config.TableConfig(
            name="feature2_table",
            vocabulary_size=VOCABULARY_SIZE,
            embedding_dim=FEATURE2_EMBEDDING_OUTPUT_DIM,
            placement=placement,
        )

        feature_group = {}
        feature_group["feature1"] = config.FeatureConfig(
            name="feature1",
            table=feature1_table,
            input_shape=(BATCH_SIZE_PER_CORE, sequence_length),
            output_shape=(BATCH_SIZE_PER_CORE, FEATURE1_EMBEDDING_OUTPUT_DIM),
        )
        feature_group["feature2"] = config.FeatureConfig(
            name="feature2",
            table=feature2_table,
            input_shape=(BATCH_SIZE_PER_CORE, sequence_length),
            output_shape=(BATCH_SIZE_PER_CORE, FEATURE2_EMBEDDING_OUTPUT_DIM),
        )
        return {"feature_group": feature_group}

    def create_inputs_weights_and_labels(
        self, batch_size, input_type, feature_configs, backend=None
    ):
        backend = backend or keras.backend.backend()

        if input_type == "dense":

            def create_tensor(feature_config, op):
                sequence_length = feature_config.input_shape[-1]
                return op((batch_size, sequence_length))

        elif input_type == "ragged":

            def create_tensor(feature_config, op):
                sequence_length = feature_config.input_shape[-1]
                row_lengths = [
                    1 + (i % sequence_length) for i in range(batch_size)
                ]
                total_length = sum(row_lengths)
                return tf.RaggedTensor.from_row_lengths(
                    op((total_length,)), row_lengths
                )

        elif input_type == "sparse" and backend == "tensorflow":

            def create_tensor(feature_config, op):
                sequence_length = feature_config.input_shape[-1]
                indices = [[i, i % sequence_length] for i in range(batch_size)]
                return tf.sparse.reorder(
                    tf.SparseTensor(
                        indices,
                        op((batch_size,)),
                        (batch_size, sequence_length),
                    )
                )

        elif input_type == "sparse" and backend == "jax":

            def create_tensor(feature_config, op):
                sequence_length = feature_config.input_shape[-1]
                indices = [[i, i % sequence_length] for i in range(batch_size)]
                return jax_sparse.BCOO(
                    (jnp.asarray(op((batch_size,))), jnp.asarray(indices)),
                    shape=(batch_size, sequence_length),
                    unique_indices=True,
                )

        else:
            raise ValueError(f"Unsupported type: {input_type}")

        inputs = keras.tree.map_structure(
            functools.partial(
                create_tensor, op=lambda shape: np.ones(shape, dtype="int32")
            ),
            feature_configs,
        )
        weights = keras.tree.map_structure(
            functools.partial(
                create_tensor,
                op=lambda shape: np.random.uniform(size=shape).astype(
                    np.float32
                ),
            ),
            feature_configs,
        )
        labels = keras.tree.map_structure(
            lambda fc: np.ones(
                (batch_size,) + fc.output_shape[1:], dtype=np.float32
            ),
            feature_configs,
        )
        return inputs, weights, labels

    @parameterized.named_parameters(
        [
            (f"{input_type}_{placement}", input_type, placement)
            for input_type in ("dense", "ragged", "sparse")
            for placement in ("auto", "default_device", "sparsecore")
        ]
    )
    def test_basics(self, input_type, placement):
        if keras.backend.backend() == "jax":
            if input_type == "sparse":
                self.skipTest("Sparse inputs not supported on JAX.")
            if input_type == "ragged" and (
                placement == "default_device" or not self.on_tpu
            ):
                self.skipTest(
                    "Ragged inputs not supported on JAX with default device "
                    "placement."
                )
        elif keras.backend.backend() != "tensorflow":
            if input_type in ("ragged", "sparse"):
                self.skipTest(
                    f"{input_type} not supported on {keras.backend.backend()}"
                )

        if (
            placement == "default_device"
            and self.on_tpu
            and input_type in ("ragged", "sparse")
        ):
            self.skipTest("Ragged and sparse are not compilable on TPU.")

        batch_size = self._strategy.num_replicas_in_sync * BATCH_SIZE_PER_CORE
        feature_configs = self.get_embedding_config(input_type, placement)
        inputs, weights, _ = self.create_inputs_weights_and_labels(
            batch_size, input_type, feature_configs
        )

        if placement == "sparsecore" and not self.on_tpu:
            with self.assertRaisesRegex(Exception, "sparsecore"):
                with self._strategy.scope():
                    embedding.DistributedEmbedding(feature_configs)
            return

        with self._strategy.scope():
            layer = embedding.DistributedEmbedding(feature_configs)

        if keras.backend.backend() == "jax":
            preprocessed_inputs = layer.preprocess(inputs, weights)
            res = layer(preprocessed_inputs)
        else:
            res = self.run_with_strategy(layer.__call__, inputs, weights)

        if placement == "default_device" or not self.on_tpu:
            # verify sublayers and variables are tracked
            self.assertLen(layer._flatten_layers(include_self=False), 2)
            self.assertLen(layer.trainable_variables, 2)
            self.assertEqual(
                layer.trainable_variables[0].shape,
                (VOCABULARY_SIZE, FEATURE1_EMBEDDING_OUTPUT_DIM),
            )
            self.assertEqual(
                layer.trainable_variables[1].shape,
                (VOCABULARY_SIZE, FEATURE2_EMBEDDING_OUTPUT_DIM),
            )

        self.assertEqual(
            res["feature_group"]["feature1"].shape,
            (batch_size, FEATURE1_EMBEDDING_OUTPUT_DIM),
        )
        self.assertEqual(
            res["feature_group"]["feature2"].shape,
            (batch_size, FEATURE2_EMBEDDING_OUTPUT_DIM),
        )

    @parameterized.named_parameters(
        ("dense", "dense", False),
        ("dense_weights", "dense", True),
        ("ragged", "ragged", False),
        ("ragged_weights", "ragged", True),
        ("sparse", "sparse", False),
        ("sparse_weights", "sparse", True),
    )
    def test_model_fit(self, input_type, use_weights):
        if keras.backend.backend() == "jax":
            if input_type == "ragged" and (
                self.placement == "default" or not self.on_tpu
            ):
                self.skipTest(
                    f"TODO {input_type} inputs on JAX with default placement."
                )
            if input_type == "sparse":
                self.skipTest("TODO sparse inputs on JAX.")
        elif keras.backend.backend() != "tensorflow":
            if input_type in ("ragged", "sparse"):
                self.skipTest(
                    f"{input_type} not supported on {keras.backend.backend()}"
                )

        batch_size = self._strategy.num_replicas_in_sync * BATCH_SIZE_PER_CORE
        feature_configs = self.get_embedding_config(input_type, self.placement)
        train_inputs, train_weights, train_labels = (
            self.create_inputs_weights_and_labels(
                batch_size, input_type, feature_configs, backend="tensorflow"
            )
        )
        test_inputs, test_weights, test_labels = (
            self.create_inputs_weights_and_labels(
                batch_size, input_type, feature_configs, backend="tensorflow"
            )
        )

        if use_weights:
            train_model_inputs = (train_inputs, train_weights)
            test_model_inputs = (test_inputs, test_weights)
        else:
            train_model_inputs = train_inputs
            test_model_inputs = test_inputs

        train_dataset = tf.data.Dataset.from_tensors(
            (train_model_inputs, train_labels)
        )
        test_dataset = tf.data.Dataset.from_tensors(
            (test_model_inputs, test_labels)
        )

        with self._strategy.scope():
            layer = embedding.DistributedEmbedding(feature_configs)

        if keras.backend.backend() == "jax":
            # Set global distribution to ensure optimizer variables are
            # replicated across all devices by default.
            keras.distribution.set_distribution(
                keras.distribution.DataParallel()
            )

            # Call preprocess on dataset inputs/weights.
            def preprocess(inputs_and_labels):
                # Extract inputs, weights and labels.
                weights = None
                inputs, labels = inputs_and_labels
                labels = keras.tree.map_structure(lambda x: x.numpy(), labels)
                if use_weights:
                    inputs, weights = inputs
                preprocessed = layer.preprocess(inputs, weights, training=True)
                return preprocessed, labels

            # Create a dataset generator that applies the preprocess function.
            # We need to create an intermediary tf_dataset to avoid
            # self-references in the generator. Repeat data so we can shard
            # across devices.
            tf_train_dataset = train_dataset.repeat(16)

            def train_dataset_generator():
                for inputs in iter(tf_train_dataset):
                    yield preprocess(inputs)

            train_dataset = train_dataset_generator()

            tf_test_dataset = test_dataset.repeat(16)

            def test_dataset_generator():
                for inputs in iter(tf_test_dataset):
                    yield preprocess(inputs)

            test_dataset = test_dataset_generator()
            model = keras.Sequential([layer])
        else:
            train_dataset = self._strategy.experimental_distribute_dataset(
                train_dataset,
                options=tf.distribute.InputOptions(
                    experimental_fetch_to_device=False
                ),
            )
            keras_inputs = keras.tree.map_structure(
                lambda fc: keras.layers.Input(
                    fc.input_shape[1:],
                    dtype="int32",
                    sparse=input_type == "sparse",
                    ragged=input_type == "ragged",
                ),
                feature_configs,
            )
            if use_weights:
                keras_weights = keras.tree.map_structure(
                    lambda fc: keras.layers.Input(
                        fc.input_shape[1:],
                        dtype="float32",
                        sparse=input_type == "sparse",
                        ragged=input_type == "ragged",
                    ),
                    feature_configs,
                )
                keras_model_inputs = (keras_inputs, keras_weights)
            else:
                keras_weights = None
                keras_model_inputs = keras_inputs

            keras_model_outputs = layer(keras_inputs, keras_weights)
            model = keras.Model(
                inputs=keras_model_inputs, outputs=keras_model_outputs
            )

        with self._strategy.scope():
            model.compile(optimizer="adam", loss="mse")

            model_inputs, _ = next(iter(test_dataset))
            test_output_before = self.run_with_strategy(
                model.__call__, model_inputs
            )

            model.fit(train_dataset, epochs=1)

            test_output_after = self.run_with_strategy(
                model.__call__, model_inputs
            )

        # Verify that the embedding has actually trained.
        for before, after in zip(
            keras.tree.flatten(test_output_before),
            keras.tree.flatten(test_output_after),
        ):
            self.assertNotAllClose(before, after)

    @parameterized.named_parameters(
        [
            (
                (
                    f"{combiner}_{input_type}_{input_rank}d"
                    f"{'_weights' if use_weights else ''}"
                    f"{'_jit' if jit_compile else ''}"
                ),
                combiner,
                input_type,
                input_rank,
                use_weights,
                jit_compile,
            )
            for combiner in ("sum", "mean", "sqrtn")
            for input_type, input_rank in (
                ("dense", 1),
                ("dense", 2),
                ("ragged", 2),
                ("sparse", 2),
            )
            for use_weights in (False, True)
            for jit_compile in (False, True)
        ]
    )
    def test_correctness(
        self, combiner, input_type, input_rank, use_weights, jit_compile
    ):
        if keras.backend.backend() == "jax":
            if input_type == "ragged" and (
                self.placement == "default" or not self.on_tpu
            ):
                self.skipTest(
                    f"TODO {input_type} inputs on JAX with default placement."
                )
            if input_type == "sparse":
                self.skipTest(f"TODO {input_type} inputs on JAX.")
        elif keras.backend.backend() == "tensorflow":
            if input_type == "ragged" and jit_compile:
                self.skipTest("TODO compilable ragged on TensorFlow.")
            if input_type == "sparse" and jit_compile:
                self.skipTest("TF SparseTensor ops are not jit compilable.")
        else:
            if input_type in ("ragged", "sparse"):
                self.skipTest(
                    f"{input_type} not supported on {keras.backend.backend()}"
                )
            if jit_compile:
                self.skipTest(
                    f"jit_compile not supported on {keras.backend.backend()}"
                )

        table = config.TableConfig(
            name="table",
            vocabulary_size=VOCABULARY_SIZE,
            embedding_dim=EMBEDDING_OUTPUT_DIM,
            combiner=combiner,
            placement=self.placement,
        )

        sequence_length = 1 if input_type == "dense" else SEQUENCE_LENGTH
        feature_config = config.FeatureConfig(
            name="feature",
            table=table,
            input_shape=(BATCH_SIZE_PER_CORE, sequence_length),
            output_shape=(BATCH_SIZE_PER_CORE, EMBEDDING_OUTPUT_DIM),
        )

        batch_size = self._strategy.num_replicas_in_sync * BATCH_SIZE_PER_CORE
        num_repeats = batch_size // 2
        if input_type == "dense" and input_rank == 1:
            inputs = keras.ops.convert_to_tensor([2, 3] * num_repeats)
            weights = keras.ops.convert_to_tensor([1.0, 2.0] * num_repeats)
        elif input_type == "dense" and input_rank == 2:
            inputs = keras.ops.convert_to_tensor([[2], [3]] * num_repeats)
            weights = keras.ops.convert_to_tensor([[1.0], [2.0]] * num_repeats)
        elif input_type == "ragged" and input_rank == 2:
            inputs = tf.ragged.constant([[1], [2, 3, 4, 5]] * num_repeats)
            weights = tf.ragged.constant(
                [[1.0], [1.0, 2.0, 3.0, 4.0]] * num_repeats
            )
        elif input_type == "sparse" and input_rank == 2:
            indices = []
            for i in range(num_repeats):
                indices.append([i * 2, 0])
                indices.append([i * 2 + 1, 0])
                indices.append([i * 2 + 1, 1])
                indices.append([i * 2 + 1, 2])
                indices.append([i * 2 + 1, 3])
            if keras.backend.backend() == "tensorflow":
                inputs = tf.sparse.reorder(
                    tf.SparseTensor(
                        indices,
                        [1, 2, 3, 4, 5] * num_repeats,
                        dense_shape=(batch_size, 4),
                    )
                )
                weights = tf.sparse.reorder(
                    tf.SparseTensor(
                        indices,
                        [1.0, 1.0, 2.0, 3.0, 4.0] * num_repeats,
                        dense_shape=(batch_size, 4),
                    )
                )
            elif keras.backend.backend() == "jax":
                inputs = jax_sparse.BCOO(
                    (
                        jnp.asarray([1, 2, 3, 4, 5] * num_repeats),
                        jnp.asarray(indices),
                    ),
                    shape=(batch_size, 4),
                    unique_indices=True,
                )
                weights = jax_sparse.BCOO(
                    (
                        jnp.asarray([1.0, 1.0, 2.0, 3.0, 4.0] * num_repeats),
                        jnp.asarray(indices),
                    ),
                    shape=(batch_size, 4),
                    unique_indices=True,
                )
            else:
                raise ValueError(
                    f"Unsupported backend: {keras.backend.backend()}"
                )
        else:
            raise ValueError(
                f"Unsupported type/rank combination: {input_type} {input_rank}d"
            )

        if not use_weights:
            weights = None

        with self._strategy.scope():
            layer = embedding.DistributedEmbedding(feature_config)

        if keras.backend.backend() == "jax":
            preprocessed = layer.preprocess(inputs, weights)
            if jit_compile:
                # Determine explicit shardings/layouts for jit compilation
                # (required for sparsecore computations).
                trainable_layouts = keras.tree.map_structure(
                    lambda x: x.value.layout, layer.trainable_variables
                )
                non_trainable_layouts = keras.tree.map_structure(
                    lambda x: x.value.layout, layer.non_trainable_variables
                )
                # Input/output data involved in sparsecore operations are
                # sharded across all sparse-core-capable devices.
                jax_mesh = jax.sharding.Mesh(jax.devices(), ["sc_data"])
                sc_data_sharding = jax.sharding.NamedSharding(
                    jax_mesh, jax.sharding.PartitionSpec(["sc_data"])
                )
                preprocessed_layouts = keras.tree.map_structure(
                    lambda _: sc_data_sharding, preprocessed
                )
                output_layouts = keras.tree.map_structure(
                    lambda _: sc_data_sharding, inputs
                )
                res, _ = jax.jit(
                    layer.stateless_call,
                    in_shardings=(
                        trainable_layouts,
                        non_trainable_layouts,
                        preprocessed_layouts,
                    ),
                    out_shardings=(
                        output_layouts,
                        non_trainable_layouts,
                    ),
                )(
                    layer.trainable_variables,
                    layer.non_trainable_variables,
                    preprocessed,
                )
            else:
                res = self.run_with_strategy(layer.__call__, preprocessed)
        else:
            res = self.run_with_strategy(
                layer.__call__, inputs, weights, jit_compile=jit_compile
            )

        self.assertEqual(res.shape, (batch_size, EMBEDDING_OUTPUT_DIM))

        tables = layer.get_embedding_tables()
        emb = tables["table"]

        if input_type == "dense":
            if combiner == "sum" and use_weights:
                expected = [emb[2], emb[3] * 2.0]
            else:
                expected = [emb[2], emb[3]]
        elif input_type in ("ragged", "sparse"):
            if use_weights:
                expected = [
                    emb[1],
                    emb[2] + emb[3] * 2.0 + emb[4] * 3.0 + emb[5] * 4.0,
                ]
            else:
                expected = [emb[1], emb[2] + emb[3] + emb[4] + emb[5]]

            if combiner == "mean":
                expected[1] /= 10.0 if use_weights else 4.0
            if combiner == "sqrtn":
                expected[1] /= (
                    math.sqrt(30.0) if use_weights else math.sqrt(4.0)
                )
        else:
            raise ValueError(f"Unsupported type: {input_type}")

        expected = expected * num_repeats

        self.assertAllClose(res, expected)

    def test_shared_table(self):
        table1 = config.TableConfig(
            name="table1",
            vocabulary_size=VOCABULARY_SIZE,
            embedding_dim=EMBEDDING_OUTPUT_DIM,
            placement=self.placement,
        )

        embedding_config = {
            "feature1": config.FeatureConfig(
                name="feature1",
                table=table1,
                input_shape=(BATCH_SIZE_PER_CORE, 1),
                output_shape=(BATCH_SIZE_PER_CORE, EMBEDDING_OUTPUT_DIM),
            ),
            "feature2": config.FeatureConfig(
                name="feature2",
                table=table1,
                input_shape=(BATCH_SIZE_PER_CORE, 1),
                output_shape=(BATCH_SIZE_PER_CORE, EMBEDDING_OUTPUT_DIM),
            ),
            "feature3": config.FeatureConfig(
                name="feature3",
                table=table1,
                input_shape=(BATCH_SIZE_PER_CORE, 1),
                output_shape=(BATCH_SIZE_PER_CORE, EMBEDDING_OUTPUT_DIM),
            ),
        }

        batch_size = self._strategy.num_replicas_in_sync * BATCH_SIZE_PER_CORE
        inputs, _, _ = self.create_inputs_weights_and_labels(
            batch_size, "dense", embedding_config
        )

        with self._strategy.scope():
            layer = embedding.DistributedEmbedding(embedding_config)

        res = self.run_with_strategy(layer.__call__, inputs)

        if self.placement == "default_device":
            self.assertLen(layer._flatten_layers(include_self=False), 1)
            self.assertLen(layer.trainable_variables, 1)

        self.assertEqual(
            res["feature1"].shape, (batch_size, EMBEDDING_OUTPUT_DIM)
        )
        self.assertEqual(
            res["feature2"].shape, (batch_size, EMBEDDING_OUTPUT_DIM)
        )
        self.assertEqual(
            res["feature3"].shape, (batch_size, EMBEDDING_OUTPUT_DIM)
        )

    def test_save_load_model(self):
        batch_size = self._strategy.num_replicas_in_sync * BATCH_SIZE_PER_CORE
        feature_configs = self.get_embedding_config("dense", self.placement)
        inputs, _, _ = self.create_inputs_weights_and_labels(
            batch_size, "dense", feature_configs
        )

        keras_inputs = keras.tree.map_structure(
            lambda fc: keras.layers.Input(fc.input_shape[1:], dtype="int32"),
            feature_configs,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "model.keras")

            with self._strategy.scope():
                layer = embedding.DistributedEmbedding(feature_configs)
                keras_outputs = layer(keras_inputs)
                model = keras.Model(inputs=keras_inputs, outputs=keras_outputs)

                output_before = self.run_with_strategy(model.__call__, inputs)
                model.save(path)

            with self._strategy.scope():
                reloaded_model = keras.models.load_model(path)
                output_after = self.run_with_strategy(
                    reloaded_model.__call__, inputs
                )

        if self.placement == "sparsecore":
            self.skipTest("TODO table reloading.")

        for before, after in zip(
            keras.tree.flatten(output_before), keras.tree.flatten(output_after)
        ):
            self.assertAllClose(before, after)
