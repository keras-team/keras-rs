import collections
from typing import Any, Optional, Sequence, Union

import keras

from keras_rs.src import types
from keras_rs.src.layers.embedding import distributed_embedding_config
from keras_rs.src.layers.embedding import embed_reduce
from keras_rs.src.utils import keras_utils

FeatureConfig = distributed_embedding_config.FeatureConfig
TableConfig = distributed_embedding_config.TableConfig
EmbedReduce = embed_reduce.EmbedReduce


SUPPORTED_PLACEMENTS = ("auto", "default_device", "sparsecore")


PlacementAndPath = collections.namedtuple(
    "PlacementAndPath", ["placement", "path"]
)


class DistributedEmbedding(keras.layers.Layer):
    """DistributedEmbedding, a layer for accelerated large embedding lookups.

    ---

    ## Note: `DistributedEmbedding` is in Preview.

    ---

    ## Configuration

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
    )
    table2 = keras_rs.layers.TableConfig(
        name="table2",
        vocabulary_size=TABLE2_VOCABULARY_SIZE,
        embedding_dim=TABLE2_EMBEDDING_SIZE,
    )

    feature1 = keras_rs.layers.FeatureConfig(
        name="feature1",
        table=table1,
        input_shape=(PER_REPLICA_BATCH_SIZE,),
        output_shape=(PER_REPLICA_BATCH_SIZE, TABLE1_EMBEDDING_SIZE),
    )
    feature2 = keras_rs.layers.FeatureConfig(
        name="feature2",
        table=table2,
        input_shape=(PER_REPLICA_BATCH_SIZE,),
        output_shape=(PER_REPLICA_BATCH_SIZE, TABLE2_EMBEDDING_SIZE),
    )

    feature_configs = {
        "feature1": feature1,
        "feature2": feature2,
    }

    embedding = keras_rs.layers.DistributedEmbedding(feature_configs)
    ```

    ## Optimizers

    Each embedding table within `DistributedEmbedding` uses its own optimizer
    for training, which is independent from the optimizer set on the model via
    `model.compile()`.

    Note that not all optimizers are supported. Currently, the following are
    always supported (i.e. on all backends and accelerators):

    - `keras.optimizers.Adagrad`
    - `keras.optimizers.SGD`

    Additionally, not all parameters of the optimizers are supported (e.g. the
    `nesterov` option of `SGD`). An error is raised when an unsupported
    optimizer or an unsupported optimizer parameter is used.

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
        table_stacking: Union[
            str, Sequence[str], Sequence[Sequence[str]]
        ] = "auto",
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
          - `flatten`
          - `pack_sequence_as` `self._placement_to_path_to_feature_config`
        - go from the two-level structure to the deeply nested structure:
         - `assert_same_struct` as `self._placement_to_path_to_feature_config`
         - `flatten`
         - `pack_sequence_as` `self._feature_configs`

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
            placement_and_path: PlacementAndPath, input_shape: types.Shape
        ) -> None:
            placement_to_path_to_input_shape[placement_and_path.placement][
                placement_and_path.path
            ] = input_shape

        keras.tree.map_structure_up_to(
            self._feature_configs,
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

    def call(
        self,
        inputs: types.Nested[types.Tensor],
        weights: Optional[types.Nested[types.Tensor]] = None,
        training: bool = False,
    ) -> types.Nested[types.Tensor]:
        """Lookup features in embedding tables and apply reduction.

        Args:
            inputs: A nested structure of 2D tensors to embed and reduce. The
                structure must be the same as the `feature_configs` passed
                during construction.
            weights: An optional nested structure of 2D tensors of weights to
               apply before reduction. When present, the structure must be the
               same as `inputs` and the shapes must match.

        Returns:
            A nested structure of dense 2D tensors, which are the reduced
            embeddings from the passed features. The structure is the same as
            `inputs`.
        """

        # Verify input structure.
        keras.tree.assert_same_structure(self._feature_configs, inputs)

        # Go from deeply nested structure of inputs to flat inputs.
        flat_inputs = keras.tree.flatten(inputs)

        # Go from flat to nested dict placement -> path -> input.
        placement_to_path_to_inputs = keras.tree.pack_sequence_as(
            self._placement_to_path_to_feature_config, flat_inputs
        )

        if weights is not None:
            # Same for weights if present.
            keras.tree.assert_same_structure(self._feature_configs, weights)
            flat_weights = keras.tree.flatten(weights)
            placement_to_path_to_weights = keras.tree.pack_sequence_as(
                self._placement_to_path_to_feature_config, flat_weights
            )
        else:
            # Populate keys for weights.
            placement_to_path_to_weights = {
                k: None for k in placement_to_path_to_inputs
            }

        placement_to_path_to_outputs = {}

        # Call for features placed on "sparsecore".
        if "sparsecore" in placement_to_path_to_inputs:
            placement_to_path_to_outputs["sparsecore"] = self._sparsecore_call(
                placement_to_path_to_inputs["sparsecore"],
                placement_to_path_to_weights["sparsecore"],
                training,
            )

        # Call for features placed on "default_device".
        if "default_device" in placement_to_path_to_inputs:
            placement_to_path_to_outputs["default_device"] = (
                self._default_device_call(
                    placement_to_path_to_inputs["default_device"],
                    placement_to_path_to_weights["default_device"],
                )
            )

        # Verify output structure.
        keras.tree.assert_same_structure(
            self._placement_to_path_to_feature_config,
            placement_to_path_to_outputs,
        )

        # Go from placement -> path -> output to flat outputs.
        flat_outputs = keras.tree.flatten(placement_to_path_to_outputs)

        # Go from flat outputs to deeply nested structure.
        return keras.tree.pack_sequence_as(self._feature_configs, flat_outputs)

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
        feature_configs: dict[str, Union[FeatureConfig]],
        table_stacking: Union[str, Sequence[Sequence[str]]],
    ) -> None:
        del table_stacking
        table_to_embedding_layer: dict[TableConfig, EmbedReduce] = {}
        self._default_device_embedding_layers: dict[str, EmbedReduce] = {}

        for path, feature_config in feature_configs.items():
            if feature_config.table in table_to_embedding_layer:
                self._default_device_embedding_layers[path] = (
                    table_to_embedding_layer[feature_config.table]
                )
            else:
                embedding_layer = EmbedReduce(
                    name=feature_config.table.name,
                    input_dim=feature_config.table.vocabulary_size,
                    output_dim=feature_config.table.embedding_dim,
                    embeddings_initializer=feature_config.table.initializer,
                    combiner=feature_config.table.combiner,
                )
                table_to_embedding_layer[feature_config.table] = embedding_layer
                self._default_device_embedding_layers[path] = embedding_layer

    def _default_device_build(
        self, input_shapes: dict[str, types.Shape]
    ) -> None:
        for path, input_shape in input_shapes.items():
            embedding_layer = self._default_device_embedding_layers[path]
            if not embedding_layer.built:
                embedding_layer.build(input_shape)

    def _default_device_call(
        self,
        inputs: dict[str, types.Tensor],
        weights: Optional[dict[str, types.Tensor]] = None,
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
                self._default_device_embedding_layers[path].embeddings
            )
        return tables

    def _has_sparsecore(self) -> bool:
        return False

    def _sparsecore_init(
        self,
        feature_configs: dict[str, FeatureConfig],
        table_stacking: Union[str, Sequence[Sequence[str]]],
    ) -> None:
        del feature_configs, table_stacking
        raise self._unsupported_placement_error("sparsecore")

    def _sparsecore_build(self, input_shapes: dict[str, types.Shape]) -> None:
        del input_shapes
        raise self._unsupported_placement_error("sparsecore")

    def _sparsecore_call(
        self,
        inputs: dict[str, types.Tensor],
        weights: Optional[dict[str, types.Tensor]] = None,
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
        # Mapping from `TableConfig` to index in `table_config_dicts`.
        table_config_indices: dict[TableConfig, int] = {}

        def serialize_feature_config(
            feature_config: FeatureConfig,
        ) -> dict[str, Any]:
            # Note that for consistency with the contract of `get_config`, the
            # returned dict contains the serialized `TableConfig` in the "table"
            # key.
            feature_config_dict = feature_config.get_config()

            if feature_config.table not in table_config_indices:
                # Save the serialized `TableConfig` the first time we see it and
                # remember its index.
                table_config_indices[feature_config.table] = len(
                    table_config_dicts
                )
                table_config_dicts.append(feature_config_dict["table"])

            # Replace the serialized `TableConfig` with its index.
            feature_config_dict["table"] = table_config_indices[
                feature_config.table
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
        table_configs: list[Optional[TableConfig]] = [None] * len(
            table_config_dicts
        )

        def deserialize_feature_config(
            feature_config_dict: dict[str, Any],
        ) -> Optional[FeatureConfig]:
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
