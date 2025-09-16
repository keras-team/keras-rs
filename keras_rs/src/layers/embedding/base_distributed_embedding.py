import collections
import dataclasses
import importlib.util
import typing
from typing import Any, Sequence

import keras
import numpy as np
from keras.src import backend

from keras_rs.src import types
from keras_rs.src.layers.embedding import distributed_embedding_config
from keras_rs.src.layers.embedding import embed_reduce
from keras_rs.src.utils import keras_utils

FeatureConfig = distributed_embedding_config.FeatureConfig
TableConfig = distributed_embedding_config.TableConfig
EmbedReduce = embed_reduce.EmbedReduce


SUPPORTED_PLACEMENTS = ("auto", "default_device", "sparsecore")


@dataclasses.dataclass(eq=True, unsafe_hash=True, order=True)
class PlacementAndPath:
    placement: str
    path: str


def _ragged_to_dense_inputs(
    inputs: Any, weights: Any | None = None, dense_row_length: int | None = None
) -> Any:
    """Converts a ragged set of inputs and weights to dense.

    If inputs are ragged and weights are `None`, will create a dense set of
    weights to mask out the new padded values.

    If inputs are not ragged, returns the original `inputs` and `weights`
    unmodified.

    Args:
        inputs: The inputs array.
        weights: The optional weights array.
        dense_row_length: The output dense row length.  If None, uses the length
             of the longest row of the input.

    Returns:
        Tuple of new (inputs, weights).  If the input is a ragged array, returns
        dense numpy arrays. Otherwise, returns the original input and weights.
    """
    x = inputs
    w = weights
    # tf.Ragged or other .numpy()-able types.
    if hasattr(x, "numpy") and callable(getattr(x, "numpy")):
        x = x.numpy()

    # Ragged numpy array to dense numpy array.
    if isinstance(x, np.ndarray) and len(x) > 0 and x.dtype == np.ndarray:
        # Maybe convert weights to numpy.
        if (
            w is not None
            and hasattr(w, "numpy")
            and callable(getattr(w, "numpy"))
        ):
            w = w.numpy()

        if dense_row_length is None:
            # Use length of longest row.
            dense_row_length = max([len(row) for row in x])

        output = np.zeros((len(x), dense_row_length), dtype=x[0].dtype)
        for i, row in enumerate(x):
            output[i, : len(row)] = row

        output_weights = np.zeros((len(x), dense_row_length), dtype=np.float32)
        if w is None:
            for i, row in enumerate(x):
                output_weights[i, : len(row)] = 1.0
        else:
            for i, row in enumerate(w):
                output_weights[i, : len(row)] = row

        return output, output_weights

    # Convert symbolic ragged/sparse keras tensors to dense tensors.
    if isinstance(x, keras.KerasTensor) and (x.ragged or x.sparse):
        inputs = keras.ops.convert_to_tensor(x, ragged=False)
        weights = keras.ops.convert_to_tensor(x, dtype="float32", ragged=False)

    # If not a ragged array, return the original, unmodified.
    return inputs, weights


class DistributedEmbedding(keras.layers.Layer):
    """DistributedEmbedding, a layer for accelerated large embedding lookups.

    ---

    ## Note: `DistributedEmbedding` is in Preview.

    ---

    `DistributedEmbedding` is a layer optimized for TPU chips with SparseCore
    and can dramatically improve the speed of embedding lookups and embedding
    training. It works by combining multiple lookups into one invocation, and by
    sharding the embedding tables across the available chips. Note that one will
    only see performance benefits for embedding tables that are large enough to
    to require sharding because they don't fit on a single chip. More details
    are provided in the "Placement" section below.

    On other hardware, GPUs, CPUs and TPUs without SparseCore,
    `DistributedEmbedding` provides the same API without any specific
    acceleration. No particular distribution scheme is applied besides the one
    set via `keras.distribution.set_distribution`.

    `DistributedEmbedding` embeds sequences of inputs and reduces them to a
    single embedding by applying a configurable combiner function.

    ### Configuration

    #### Features and tables

    A `DistributedEmbedding` embedding layer is configured via a set of
    `keras_rs.layers.FeatureConfig` objects, which themselves refer to
    `keras_rs.layers.TableConfig` objects.

    - `TableConfig` defines an embedding table with parameters such as its
      vocabulary size, embedding dimension, as well as a combiner for reduction
      and optimizer for training.
    - `FeatureConfig` defines what input features the `DistributedEmbedding`
      will handle and which embedding table to use. Note that multiple features
      can use the same embedding table.

    ```python
    table1 = keras_rs.layers.TableConfig(
        name="table1",
        vocabulary_size=TABLE1_VOCABULARY_SIZE,
        embedding_dim=TABLE1_EMBEDDING_SIZE,
        placement="auto",
    )
    table2 = keras_rs.layers.TableConfig(
        name="table2",
        vocabulary_size=TABLE2_VOCABULARY_SIZE,
        embedding_dim=TABLE2_EMBEDDING_SIZE,
        placement="auto",
    )

    feature1 = keras_rs.layers.FeatureConfig(
        name="feature1",
        table=table1,
        input_shape=(GLOBAL_BATCH_SIZE,),
        output_shape=(GLOBAL_BATCH_SIZE, TABLE1_EMBEDDING_SIZE),
    )
    feature2 = keras_rs.layers.FeatureConfig(
        name="feature2",
        table=table2,
        input_shape=(GLOBAL_BATCH_SIZE,),
        output_shape=(GLOBAL_BATCH_SIZE, TABLE2_EMBEDDING_SIZE),
    )

    feature_configs = {
        "feature1": feature1,
        "feature2": feature2,
    }

    embedding = keras_rs.layers.DistributedEmbedding(feature_configs)
    ```

    #### Optimizers

    Each embedding table within `DistributedEmbedding` uses its own optimizer
    for training, which is independent from the optimizer set on the model via
    `model.compile()`.

    Note that not all optimizers are supported. Currently, the following are
    supported on all backends and accelerators:

    - `keras.optimizers.Adagrad`
    - `keras.optimizers.Adam`
    - `keras.optimizers.Ftrl`
    - `keras.optimizers.SGD`

    Also, not all parameters of the optimizers are supported (e.g. the
    `nesterov` option of `SGD`). An error is raised when an unsupported
    optimizer or an unsupported optimizer parameter is used.

    #### Placement

    Each embedding table within `DistributedEmbedding` can be either placed on
    the SparseCore chip or the default device placement for the accelerator
    (e.g. HBM of the Tensor Cores on TPU). This is controlled by the `placement`
    attribute of `keras_rs.layers.TableConfig`.

    - A placement of `"sparsecore"` indicates that the table should be placed on
      the SparseCore chips. An error is raised if this option is selected and
      there are no SparseCore chips.
    - A placement of `"default_device"` indicates that the table should not be
      placed on SparseCore, even if available. Instead the table is placed on
      the device where the model normally goes, i.e. the HBM on TPUs and GPUs.
      In this case, if applicable, the table is distributed using the scheme set
      via `keras.distribution.set_distribution`. On GPUs, CPUs and TPUs without
      SparseCore, this is the only placement available, and is the one selected
      by `"auto"`.
    - A placement of `"auto"` indicates to use `"sparsecore"` if available, and
      `"default_device"` otherwise. This is the default when not specified.

    To optimize performance on TPU:

    - Tables that are so large that they need to be sharded should use the
      `"sparsecore"` placement.
    - Tables that are small enough should use `"default_device"` and should
      typically be replicated across TPUs by using the
      `keras.distribution.DataParallel` distribution option.

    ### Usage with TensorFlow on TPU with SpareCore

    #### Inputs

    In addition to `tf.Tensor`, `DistributedEmbedding` accepts `tf.RaggedTensor`
    and `tf.SparseTensor` as inputs for the embedding lookups. Ragged tensors
    must be ragged in the dimension with index 1. Note that if weights are
    passed, each weight tensor must be of the same class as the inputs for that
    particular feature and use the exact same ragged row lenghts for ragged
    tensors, and the same indices for sparse tensors. All the output of
    `DistributedEmbedding` are dense tensors.

    #### Setup

    To use `DistributedEmbedding` on TPUs with TensorFlow, one must use a
    `tf.distribute.TPUStrategy`. The `DistributedEmbedding` layer must be
    created under the `TPUStrategy`.

    ```python
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")
    topology = tf.tpu.experimental.initialize_tpu_system(resolver)
    device_assignment = tf.tpu.experimental.DeviceAssignment.build(
        topology, num_replicas=resolver.get_tpu_system_metadata().num_cores
    )
    strategy = tf.distribute.TPUStrategy(
        resolver, experimental_device_assignment=device_assignment
    )

    with strategy.scope():
        embedding = keras_rs.layers.DistributedEmbedding(feature_configs)
    ```

    #### Usage in a Keras model

    To use Keras' `model.fit()`, one must compile the model under the
    `TPUStrategy`. Then, `model.fit()`, `model.evaluate()` or `model.predict()`
    can be called directly. The Keras model takes care of running the model
    using the strategy and also automatically distributes the dataset.

    ```python
    with strategy.scope():
        embedding = keras_rs.layers.DistributedEmbedding(feature_configs)
        model = create_model(embedding)
        model.compile(loss=keras.losses.MeanSquaredError(), optimizer="adam")

    model.fit(dataset, epochs=10)
    ```

    #### Direct invocation

    `DistributedEmbedding` must be invoked via a `strategy.run` call nested in a
    `tf.function`.

    ```python
    @tf.function
    def embedding_wrapper(tf_fn_inputs, tf_fn_weights=None):
        def strategy_fn(st_fn_inputs, st_fn_weights):
            return embedding(st_fn_inputs, st_fn_weights)

        return strategy.run(strategy_fn, args=(tf_fn_inputs, tf_fn_weights)))

    embedding_wrapper(my_inputs, my_weights)
    ```

    When using a dataset, the dataset must be distributed. The iterator can then
    be passed to the `tf.function` that uses `strategy.run`.

    ```python
    dataset = strategy.experimental_distribute_dataset(dataset)

    @tf.function
    def run_loop(iterator):
        def step(data):
            (inputs, weights), labels = data
            with tf.GradientTape() as tape:
                result = embedding(inputs, weights)
                loss = keras.losses.mean_squared_error(labels, result)
            tape.gradient(loss, embedding.trainable_variables)
            return result

        for _ in tf.range(4):
            result = strategy.run(step, args=(next(iterator),))

    run_loop(iter(dataset))
    ```

    ### Usage with JAX on TPU with SpareCore

    #### Setup

    To use `DistributedEmbedding` on TPUs with JAX, one must create and set a
    Keras `Distribution`.
    ```python
    distribution = keras.distribution.DataParallel(devices=jax.device("tpu"))
    keras.distribution.set_distribution(distribution)
    ```

    #### Inputs

    For JAX, inputs can either be dense tensors, or ragged (nested) NumPy
    arrays.  To enable `jit_compile = True`, one must explicitly call
    `layer.preprocess(...)` on the inputs, and then feed the preprocessed
    output to the model.  See the next section on preprocessing for details.

    Ragged input arrays must be ragged in the dimension with index 1. Note that
    if weights are passed, each weight tensor must be of the same class as the
    inputs for that particular feature and use the exact same ragged row lengths
    for ragged tensors. All the output of `DistributedEmbedding` are dense
    tensors.

    #### Preprocessing

    In JAX, SparseCore usage requires specially formatted data that depends
    on properties of the available hardware.  This data reformatting
    currently does not support jit-compilation, so must be applied _prior_
    to passing data into a model.

    Preprocessing works on dense or ragged NumPy arrays, or on tensors that are
    convertible to dense or ragged NumPy arrays like `tf.RaggedTensor`.

    One simple way to add preprocessing is to append the function to an input
    pipeline by using a python generator.
    ```python
    # Create the embedding layer.
    embedding_layer = DistributedEmbedding(feature_configs)

    # Add preprocessing to a data input pipeline.
    def preprocessed_dataset_generator(dataset):
        for (inputs, weights), labels in iter(dataset):
            yield embedding_layer.preprocess(
                inputs, weights, training=True
            ), labels

    preprocessed_train_dataset = preprocessed_dataset_generator(train_dataset)
    ```
    This explicit preprocessing stage combines the input and optional weights,
    so the new data can be passed directly into the `inputs` argument of the
    layer or model.

    **NOTE**: When working in a multi-host setting with data parallelism, the
    data needs to be sharded properly across hosts.  If the original dataset is
    of type `tf.data.Dataset`, it will need to be manually sharded _prior_ to
    applying the preprocess generator:
    ```python
    # Manually shard the dataset across hosts.
    train_dataset = distribution.distribute_dataset(train_dataset)
    distribution.auto_shard_dataset = False  # Dataset is already sharded.

    # Add a preprocessing stage to the distributed data input pipeline.
    train_dataset = preprocessed_dataset_generator(train_dataset)
    ```
    If the original dataset is _not_ a `tf.data.Dataset`, it must already be
    pre-sharded across hosts.

    #### Usage in a Keras model

    Once the global distribution is set and the input preprocessing pipeline
    is defined, model training can proceed as normal.  For example:
    ```python
    # Construct, compile, and fit the model using the preprocessed data.
    model = keras.Sequential(
      [
        embedding_layer,
        keras.layers.Dense(2),
        keras.layers.Dense(3),
        keras.layers.Dense(4),
      ]
    )
    model.compile(optimizer="adam", loss="mse", jit_compile=True)
    model.fit(preprocessed_train_dataset, epochs=10)
    ```

    #### Direct invocation

    The `DistributedEmbedding` layer can also be invoked directly.  Explicit
    preprocessing is required when used with JIT compilation.
    ```python
    # Call the layer directly.
    activations = embedding_layer(my_inputs, my_weights)

    # Call the layer with JIT compilation and explicitly preprocessed inputs.
    embedding_layer_jit = jax.jit(embedding_layer)
    preprocessed_inputs = embedding_layer.preprocess(my_inputs, my_weights)
    activations = embedding_layer_jit(preprocessed_inputs)
    ```

    Similarly, for custom training loops, preprocessing must be applied prior
    to passing the data to the JIT-compiled training step.
    ```python
    # Create an optimizer and loss function.
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)

    def loss_and_updates(trainable_variables, non_trainable_variables, x, y):
        y_pred, non_trainable_variables = model.stateless_call(
            trainable_variables, non_trainable_variables, x, training=True
        )
        loss = keras.losses.mean_squared_error(y, y_pred)
        return loss, non_trainable_variables

    grad_fn = jax.value_and_grad(loss_and_updates, has_aux=True)

    # Create a JIT-compiled training step.
    @jax.jit
    def train_step(state, x, y):
        (
          trainable_variables,
          non_trainable_variables,
          optimizer_variables,
        ) = state
        (loss, non_trainable_variables), grads = grad_fn(
            trainable_variables, non_trainable_variables, x, y
        )
        trainable_variables, optimizer_variables = optimizer.stateless_apply(
            optimizer_variables, grads, trainable_variables
        )
        return loss, (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
        )

    # Build optimizer variables.
    optimizer.build(model.trainable_variables)

    # Assemble the training state.
    trainable_variables = model.trainable_variables
    non_trainable_variables = model.non_trainable_variables
    optimizer_variables = optimizer.variables
    state = trainable_variables, non_trainable_variables, optimizer_variables

    # Training loop.
    for (inputs, weights), labels in train_dataset:
        # Explicitly preprocess the data.
        preprocessed_inputs = embedding_layer.preprocess(inputs, weights)
        loss, state = train_step(state, preprocessed_inputs, labels)
    ```

    Args:
        feature_configs: A nested structure of `keras_rs.layers.FeatureConfig`.
        table_stacking: The table stacking to use. `None` means no table
            stacking. `"auto"` means to stack tables automatically. A list of
            table names or list of lists of table names means to stack the
            tables in the inner lists together. Note that table stacking is not
            supported on older TPUs, in which case the default value of `"auto"`
            will be interpreted as no table stacking.
        **kwargs: Additional arguments to pass to the layer base class.
    """

    def __init__(
        self,
        feature_configs: types.Nested[FeatureConfig],
        *,
        table_stacking: (
            str | Sequence[str] | Sequence[Sequence[str]]
        ) = "auto",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self._init_feature_configs_structures(feature_configs)

        # Initialize for features placed on "sparsecore".
        if "sparsecore" in self._placement_to_path_to_feature_config:
            self._sparsecore_init(
                self._placement_to_path_to_feature_config["sparsecore"],
                table_stacking,
            )
        # Initialize for features placed on "default_device".
        if "default_device" in self._placement_to_path_to_feature_config:
            self._default_device_init(
                self._placement_to_path_to_feature_config["default_device"],
                table_stacking,
            )

    @keras_utils.no_automatic_dependency_tracking
    def _init_feature_configs_structures(
        self,
        feature_configs: types.Nested[FeatureConfig],
    ) -> None:
        """Initializations for efficiently transforming nested structures.

        This layer handles arbitrarily nested structures for input features, and
        therefore for outputs and feature configs. However, as an intermediary
        format we use a two-level representation with nested dicts. the top
        level dict is keyed by placement and the inner dict is keyed by path,
        with the path representing the path in the original deeply nested
        structure. Thanks to this intermediate representation, we can:
        - dispatch the inputs by placement to overridden methods
        - have backend specific implementations support only one level of
          nesting.

        This method is responsible for creating structures that allow this
        conversion to happen in a few lines of code and efficiently. The
        following attributes are created:
        - self._feature_configs: the deeply nested `FeatureConfig` instances as
          provided by user in `__init__`
        - self._feature_deeply_nested_placement_and_paths: `PlacementAndPath`
          instances in the same deeply nested structure as
          `self._feature_configs`. Needed for `build` because flatten cannot be
          used as it would expand the shape tuples.
        - self._placement_to_path_to_feature_config: `FeatureConfig` instances
          in the same two-level representation keyed by placement and then path.
          Used to go from a flat representation to the intermediate
          representation.

        With these structures in place, the steps to:
        - go from the deeply nested structure to the two-level structure are:
          - `assert_same_struct` as `self._feature_configs`
          - use `self._feature_deeply_nested_placement_and_paths` to map from
            deeply nested to two-level
        - go from the two-level structure to the deeply nested structure:
          - `assert_same_struct` as `self._placement_to_path_to_feature_config`
          - use `self._feature_deeply_nested_placement_and_paths` to locate each
            output in the two-level dicts

        Args:
            feature_configs: The deeply nested structure of `FeatureConfig` or
                `tf.tpu.experimental.embedding.FeatureConfig` as provided by the
                user.
        """
        # Needs to be assigned with `no_automatic_dependency_tracking` to not
        # alter the data structure types.
        self._feature_configs = feature_configs

        placement_and_paths: list[PlacementAndPath] = []
        paths_and_feature_configs = keras.tree.flatten_with_path(
            self._feature_configs
        )
        self._placement_to_path_to_feature_config: dict[
            str, dict[str, FeatureConfig]
        ] = {}

        # Lazily initialized.
        has_sparsecore = None

        for path, feature_config in paths_and_feature_configs:
            if isinstance(feature_config, FeatureConfig):
                placement = feature_config.table.placement
                # Resolve "auto" to an actual placement.
                if placement == "auto":
                    if has_sparsecore is None:
                        has_sparsecore = self._has_sparsecore()
                    placement = (
                        "sparsecore" if has_sparsecore else "default_device"
                    )
            else:
                # It's a `tf.tpu.experimental.embedding.FeatureConfig`.
                placement = "sparsecore"

            path = ".".join([str(e) for e in path])
            if placement not in SUPPORTED_PLACEMENTS:
                raise ValueError(
                    f"Feature '{path}' with name '{feature_config.name}' has "
                    f"unsupported placement '{placement}'."
                )
            placement_and_paths.append(PlacementAndPath(placement, path))
            if placement not in self._placement_to_path_to_feature_config:
                self._placement_to_path_to_feature_config[placement] = {}
            self._placement_to_path_to_feature_config[placement][path] = (
                feature_config
            )

        self._feature_deeply_nested_placement_and_paths = (
            keras.tree.pack_sequence_as(
                self._feature_configs, placement_and_paths
            )
        )

    def build(self, input_shapes: types.Nested[types.Shape]) -> None:
        if self.built:
            return

        self._verify_input_shapes(input_shapes)

        # Go from deeply nested structure to placement -> path -> input shape.
        placement_to_path_to_input_shape: collections.defaultdict[
            str, dict[str, types.Shape]
        ] = collections.defaultdict(dict)

        def populate_placement_to_path_to_input_shape(
            pp: PlacementAndPath, input_shape: types.Shape
        ) -> None:
            placement_to_path_to_input_shape[pp.placement][pp.path] = (
                input_shape
            )

        keras.tree.map_structure_up_to(
            self._feature_deeply_nested_placement_and_paths,
            populate_placement_to_path_to_input_shape,
            self._feature_deeply_nested_placement_and_paths,
            input_shapes,
        )

        # Build for features placed on "sparsecore".
        if "sparsecore" in placement_to_path_to_input_shape:
            self._sparsecore_build(
                placement_to_path_to_input_shape["sparsecore"]
            )

        # Build for features placed on "default_device".
        if "default_device" in placement_to_path_to_input_shape:
            self._default_device_build(
                placement_to_path_to_input_shape["default_device"]
            )

        super().build(input_shapes)

    def preprocess(
        self,
        inputs: types.Nested[types.Tensor],
        weights: types.Nested[types.Tensor] | None = None,
        training: bool = False,
    ) -> types.Nested[types.Tensor]:
        """Preprocesses and reformats the data for consumption by the model.

        For the JAX backend, converts the input data to a hardward-dependent
        format required for use with SparseCores.  Calling `preprocess`
        explicitly is only necessary to enable `jit_compile = True`.

        For non-JAX backends, preprocessing will bundle together the inputs and
        weights, and separate the inputs by device placement.  This step is
        entirely optional.

        Args:
            inputs: Ragged or dense set of sample IDs.
            weights: Optional ragged or dense set of sample weights.
            training: If true, will update internal parameters, such as
                required buffer sizes for the preprocessed data.

        Returns:
            Set of preprocessed inputs that can be fed directly into the
            `inputs` argument of the layer.
        """
        # Verify input structure.
        keras.tree.assert_same_structure(self._feature_configs, inputs)
        if weights is not None:
            keras.tree.assert_same_structure(self._feature_configs, weights)

        if not self.built:
            input_shapes = keras.tree.map_structure(
                lambda array: backend.standardize_shape(array.shape),
                inputs,
            )
            self.build(input_shapes)

        # Go from deeply nested to nested dict placement -> path -> input.
        def to_placement_to_path(
            tensors: types.Nested[types.Tensor],
        ) -> dict[str, dict[str, types.Tensor]]:
            result: dict[str, dict[str, types.Tensor]] = {
                p: dict() for p in self._placement_to_path_to_feature_config
            }

            def populate(pp: PlacementAndPath, x: types.Tensor) -> None:
                result[pp.placement][pp.path] = x

            keras.tree.map_structure(
                populate,
                self._feature_deeply_nested_placement_and_paths,
                tensors,
            )
            return result

        placement_to_path_to_inputs = to_placement_to_path(inputs)

        # Same for weights if present.
        placement_to_path_to_weights = (
            to_placement_to_path(weights) if weights is not None else None
        )

        placement_to_path_to_preprocessed: dict[
            str, dict[str, dict[str, types.Nested[types.Tensor]]]
        ] = {}

        # Preprocess for features placed on "sparsecore".
        if "sparsecore" in placement_to_path_to_inputs:
            placement_to_path_to_preprocessed["sparsecore"] = (
                self._sparsecore_preprocess(
                    placement_to_path_to_inputs["sparsecore"],
                    placement_to_path_to_weights["sparsecore"]
                    if placement_to_path_to_weights is not None
                    else None,
                    training,
                )
            )

        # Preprocess for features placed on "default_device".
        if "default_device" in placement_to_path_to_inputs:
            placement_to_path_to_preprocessed["default_device"] = (
                self._default_device_preprocess(
                    placement_to_path_to_inputs["default_device"],
                    placement_to_path_to_weights["default_device"]
                    if placement_to_path_to_weights is not None
                    else None,
                    training,
                )
            )

        # Mark inputs as preprocessed using an extra level of nesting.
        # This is necessary to detect whether inputs are already preprocessed
        # in `call`.
        output = {
            "preprocessed_inputs_per_placement": (
                placement_to_path_to_preprocessed
            )
        }
        return output

    def _is_preprocessed(
        self, inputs: types.Nested[types.Tensor | types.Shape]
    ) -> bool:
        """Checks if the input is already preprocessed."""
        return (
            isinstance(inputs, dict)
            and "preprocessed_inputs_per_placement" in inputs
        )

    def call(
        self,
        inputs: types.Nested[types.Tensor],
        weights: types.Nested[types.Tensor] | None = None,
        training: bool = False,
    ) -> types.Nested[types.Tensor]:
        """Lookup features in embedding tables and apply reduction.

        Args:
            inputs: A nested structure of 2D tensors to embed and reduce. The
                structure must be the same as the `feature_configs` passed
                during construction.  Alternatively, may consist of already
                preprocessed inputs (see `preprocess`).
            weights: An optional nested structure of 2D tensors of weights to
               apply before reduction. When present, the structure must be the
               same as `inputs` and the shapes must match.
            training: Whether we are training or evaluating the model.

        Returns:
            A nested structure of dense 2D tensors, which are the reduced
            embeddings from the passed features. The structure is the same as
            `inputs`.
        """
        preprocessed_inputs = inputs
        # Preprocess if not already done.
        if not self._is_preprocessed(inputs):
            preprocessed_inputs = self.preprocess(inputs, weights, training)

        preprocessed_inputs = typing.cast(
            dict[str, dict[str, dict[str, types.Tensor]]], preprocessed_inputs
        )
        # Placement -> path -> preprocessed inputs.
        preprocessed_inputs = preprocessed_inputs[
            "preprocessed_inputs_per_placement"
        ]

        placement_to_path_to_outputs = {}

        # Call for features placed on "sparsecore".
        if "sparsecore" in preprocessed_inputs:
            inputs_and_weights = preprocessed_inputs["sparsecore"]
            placement_to_path_to_outputs["sparsecore"] = self._sparsecore_call(
                **inputs_and_weights,
                training=training,
            )

        # Call for features placed on "default_device".
        if "default_device" in preprocessed_inputs:
            inputs_and_weights = preprocessed_inputs["default_device"]
            placement_to_path_to_outputs["default_device"] = (
                self._default_device_call(
                    **inputs_and_weights,
                    training=training,
                )
            )

        # Verify output structure.
        keras.tree.assert_same_structure(
            self._placement_to_path_to_feature_config,
            placement_to_path_to_outputs,
        )

        # Go from placement -> path -> output to deeply nested structure.
        def populate_output(pp: PlacementAndPath) -> types.Tensor:
            return placement_to_path_to_outputs[pp.placement][pp.path]

        return keras.tree.map_structure(
            populate_output, self._feature_deeply_nested_placement_and_paths
        )

    def get_embedding_tables(self) -> dict[str, types.Tensor]:
        """Return the content of the embedding tables by table name.

        The tables are keyed by the name provided in each `TableConfig`. Note
        that the returned tensors are not the actual embedding table variables
        used internally by `DistributedEmbedding`.

        Returns:
            A dictionary of table name to tensor for the embedding tables.
        """
        tables = {}
        if "sparsecore" in self._placement_to_path_to_feature_config:
            tables.update(self._sparsecore_get_embedding_tables())
        if "default_device" in self._placement_to_path_to_feature_config:
            tables.update(self._default_device_get_embedding_tables())
        return tables

    def _default_device_init(
        self,
        feature_configs: dict[str, FeatureConfig],
        table_stacking: str | Sequence[Sequence[str]],
    ) -> None:
        del table_stacking
        table_config_id_to_embedding_layer: dict[int, EmbedReduce] = {}
        self._default_device_embedding_layers: dict[str, EmbedReduce] = {}

        for path, feature_config in feature_configs.items():
            if id(feature_config.table) in table_config_id_to_embedding_layer:
                self._default_device_embedding_layers[path] = (
                    table_config_id_to_embedding_layer[id(feature_config.table)]
                )
            else:
                embedding_layer = EmbedReduce(
                    name=feature_config.table.name,
                    input_dim=feature_config.table.vocabulary_size,
                    output_dim=feature_config.table.embedding_dim,
                    embeddings_initializer=feature_config.table.initializer,
                    combiner=feature_config.table.combiner,
                )
                table_config_id_to_embedding_layer[id(feature_config.table)] = (
                    embedding_layer
                )
                self._default_device_embedding_layers[path] = embedding_layer

    def _default_device_build(
        self, input_shapes: dict[str, types.Shape]
    ) -> None:
        for path, input_shape in input_shapes.items():
            embedding_layer = self._default_device_embedding_layers[path]
            if not embedding_layer.built:
                embedding_layer.build(input_shape)

    def _default_device_preprocess(
        self,
        inputs: dict[str, types.Tensor],
        weights: dict[str, types.Tensor] | None,
        training: bool = False,
    ) -> dict[str, dict[str, types.Tensor]]:
        del training

        # NOTE: This JAX specialization is in the base layer so it is available
        #       on all platforms.  The superclass jax.DistributedEmbedding layer
        #       is currently only imported in linux_x86_64.
        if keras.backend.backend() == "jax":
            feature_configs = self._placement_to_path_to_feature_config[
                "default_device"
            ]

            # Potentially track new weights.  For ragged inputs, if we
            # densify, we will generate a dense weight tensor.
            new_weights: dict[str, types.Tensor] = {}
            use_weights = weights is not None

            # Convert any ragged inputs to dense.
            for path, config in feature_configs.items():
                feature_inputs = inputs[path]
                feature_weights = weights[path] if weights is not None else None

                feature_valence = (
                    None
                    if len(config.input_shape) <= 1
                    else config.input_shape[1]
                )
                feature_inputs, feature_weights = _ragged_to_dense_inputs(
                    feature_inputs, feature_weights, feature_valence
                )
                # Converting to ragged may have introduced a weights array.
                use_weights = use_weights or feature_weights is not None
                inputs[path] = feature_inputs
                new_weights[path] = feature_weights

            if use_weights:
                weights = new_weights

        output: dict[str, types.Tensor] = {"inputs": inputs}
        if weights is not None:
            output["weights"] = weights

        return output

    def _default_device_call(
        self,
        inputs: dict[str, types.Tensor],
        weights: dict[str, types.Tensor] | None = None,
        training: bool = False,
    ) -> dict[str, types.Tensor]:
        del training  # Unused by default.
        if weights is None:
            return {
                path: self._default_device_embedding_layers[path](x)
                for path, x in inputs.items()
            }
        else:
            return {
                path: self._default_device_embedding_layers[path](
                    x, weights[path]
                )
                for path, x in inputs.items()
            }

    def _default_device_get_embedding_tables(self) -> dict[str, types.Tensor]:
        tables = {}
        for path, feature_config in self._placement_to_path_to_feature_config[
            "default_device"
        ].items():
            tables[feature_config.table.name] = (
                self._default_device_embedding_layers[path].embeddings.value
            )
        return tables

    def _has_sparsecore(self) -> bool:
        # Explicitly check for SparseCore availability.
        # We need this check here rather than in jax/distributed_embedding.py
        # so that we can warn the user about missing dependencies.
        if keras.backend.backend() == "jax":
            # Check if SparseCores are available.
            try:
                import jax

                tpu_devices = jax.devices("tpu")
            except RuntimeError:
                # No TPUs available.
                return False

            if len(tpu_devices) > 0:
                device_kind = tpu_devices[0].device_kind
                if device_kind in ["TPU v5", "TPU v6 lite"]:
                    return True

        return False

    def _sparsecore_init(
        self,
        feature_configs: dict[str, FeatureConfig],
        table_stacking: str | Sequence[Sequence[str]],
    ) -> None:
        del feature_configs, table_stacking

        if keras.backend.backend() == "jax":
            jax_tpu_embedding_spec = importlib.util.find_spec(
                "jax_tpu_embedding"
            )
            if jax_tpu_embedding_spec is None:
                raise ImportError(
                    "Please install jax-tpu-embedding to use "
                    "DistributedEmbedding on sparsecore devices."
                )

        raise self._unsupported_placement_error("sparsecore")

    def _sparsecore_build(self, input_shapes: dict[str, types.Shape]) -> None:
        del input_shapes
        raise self._unsupported_placement_error("sparsecore")

    def _sparsecore_preprocess(
        self,
        inputs: dict[str, types.Tensor],
        weights: dict[str, types.Tensor] | None,
        training: bool = False,
    ) -> dict[str, dict[str, types.Tensor]]:
        del training
        output: dict[str, types.Tensor] = {"inputs": inputs}
        if weights is not None:
            output["weights"] = weights

        return output

    def _sparsecore_call(
        self,
        inputs: dict[str, types.Tensor],
        weights: dict[str, types.Tensor] | None = None,
        training: bool = False,
    ) -> dict[str, types.Tensor]:
        del inputs, weights, training
        raise self._unsupported_placement_error("sparsecore")

    def _sparsecore_get_embedding_tables(self) -> dict[str, types.Tensor]:
        raise self._unsupported_placement_error("sparsecore")

    def compute_output_shape(
        self, input_shapes: types.Nested[types.Shape]
    ) -> types.Nested[types.Shape]:
        self._verify_input_shapes(input_shapes)
        output_shape: types.Nested[types.Shape] = keras.tree.map_structure(
            lambda fc: fc.output_shape, self._feature_configs
        )
        return output_shape

    def get_config(self) -> dict[str, Any]:
        # Because the Keras serialization creates a tree of serialized objects,
        # it does not directly support sharing tables between feature configs.
        # We therefore serialize the tables config as a flat list and then refer
        # to them by index in each feature config.

        # The serialized `TableConfig` objects.
        table_config_dicts: list[dict[str, Any]] = []
        # Mapping from `TableConfig` id to index in `table_config_dicts`.
        table_config_id_to_index: dict[int, int] = {}

        def serialize_feature_config(
            feature_config: FeatureConfig,
        ) -> dict[str, Any]:
            # Note that for consistency with the contract of `get_config`, the
            # returned dict contains the serialized `TableConfig` in the "table"
            # key.
            feature_config_dict = feature_config.get_config()

            if id(feature_config.table) not in table_config_id_to_index:
                # Save the serialized `TableConfig` the first time we see it and
                # remember its index.
                table_config_id_to_index[id(feature_config.table)] = len(
                    table_config_dicts
                )
                table_config_dicts.append(feature_config_dict["table"])

            # Replace the serialized `TableConfig` with its index.
            feature_config_dict["table"] = table_config_id_to_index[
                id(feature_config.table)
            ]
            return feature_config_dict

        config: dict[str, Any] = super().get_config()
        config["feature_configs"] = keras.tree.map_structure(
            serialize_feature_config, self._feature_configs
        )
        config["tables"] = table_config_dicts
        if hasattr(self, "_table_stacking"):
            config["table_stacking"] = self._table_stacking
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DistributedEmbedding":
        config = config.copy()
        # We need to reconnect the `TableConfig`s to the `FeatureConfig`s.

        # The serialized `TableConfig` objects.
        table_config_dicts: list[dict[str, Any]] = config.pop("tables")
        # The deserialized `TableConfig` objects at the same indices.
        table_configs: list[TableConfig | None] = [None] * len(
            table_config_dicts
        )

        def deserialize_feature_config(
            feature_config_dict: dict[str, Any],
        ) -> FeatureConfig | None:
            # Look for a "name" attribute which is a string to detect a
            # `FeatureConfig` leaf node. If not, keep recursing.
            if "name" not in feature_config_dict or not isinstance(
                feature_config_dict["name"], str
            ):
                # Tell `traverse` to recurse.
                return None

            table_index = feature_config_dict["table"]
            # Note that for consistency with the contract of `from_config`, the
            # passed dict must contain the serialized `TableConfig` in the
            # "table" key.
            feature_config_dict["table"] = table_config_dicts[table_index]
            feature_config = FeatureConfig.from_config(feature_config_dict)
            # But then dedupe `TableConfig`s.
            if table_configs[table_index] is None:
                # Remember each new `TableConfig` we see.
                table_configs[table_index] = feature_config.table
            else:
                # And swap duplicates for the original.
                feature_config.table = table_configs[table_index]
            return feature_config

        # Because each `FeatureConfig` is serialized as a dict, we cannot use
        # `map_structure` as it would recurse in the config itself. We use
        # `traverse` instead with a function that detects leaf nodes.
        config["feature_configs"] = keras.tree.traverse(
            deserialize_feature_config, config["feature_configs"]
        )
        return cls(**config)

    def _verify_input_shapes(
        self, input_shapes: types.Nested[types.Shape]
    ) -> None:
        """Verifies that the input shapes match the ones in the feature configs.

        Args:
          input_shapes: The structure of input shapes to verify.
        """
        # Support preprocessing.
        if self._is_preprocessed(input_shapes):
            # Structure should be :
            # {
            #   placement: {
            #     inputs: {path: Any},
            #     weights: {path: Any}
            #   }
            # }
            #
            # But the `Any` values could be nested tensors with varying
            # structure, depending on hardware constraints.  This complicates
            # checking shapes via keras.tree methods.  So, assume the
            # input is a result of explicitly calling the `preprocess(...)`
            # function, in which case the structure has already been verified.
            return

        def _verify_input_shape(
            feature_config: FeatureConfig,
            input_shape: types.Shape,
        ) -> None:
            if not isinstance(input_shape, (tuple, list)) or not all(
                isinstance(d, (int, type(None))) for d in input_shape
            ):
                raise ValueError(f"Received invalid input shape {input_shape}.")
            if len(input_shape) < 1:
                raise ValueError(
                    f"Received input shape {input_shape}. Rank must be 1 or "
                    "above."
                )
            keras_utils.check_shapes_compatible(
                feature_config.input_shape, input_shape
            )

        keras.tree.map_structure_up_to(
            self._feature_configs,
            _verify_input_shape,
            self._feature_configs,
            input_shapes,
        )

    def _unsupported_placement_error(self, placement: str) -> Exception:
        return NotImplementedError(
            f"Backend '{keras.backend.backend()}' does not support the "
            f"'{placement}' placement."
        )
