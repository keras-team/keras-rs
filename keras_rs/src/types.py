"""Type definitions."""

from typing import (
    Any,
    Callable,
    Mapping,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)

import keras

"""
A tensor in any of the backends.

We do not define it explicitly to not require all the backends to be installed
and imported. The explicit definition would be:
```
Union[
  numpy.ndarray,
  tensorflow.Tensor,
  tensorflow.RaggedTensor,
  tensorflow.SparseTensor,
  tensorflow.IndexedSlices,
  jax.Array,
  jax.experimental.sparse.JAXSparse,
  torch.Tensor,
  keras.KerasTensor,
]
```
"""
Tensor = Any

Shape = Sequence[Optional[int]]
TensorShape = Shape

DType = str

ConstraintLike = Union[
    str,
    keras.constraints.Constraint,
    Type[keras.constraints.Constraint],
    Callable[[Tensor], Tensor],
]

InitializerLike = Union[
    str,
    keras.initializers.Initializer,
    Type[keras.initializers.Initializer],
    Callable[[Shape, DType], Tensor],
    Tensor,
]

RegularizerLike = Union[
    str,
    keras.regularizers.Regularizer,
    Type[keras.regularizers.Regularizer],
    Callable[[Tensor], Tensor],
]

T = TypeVar("T")
Nested = Union[
    T,
    Sequence[Union[T, "Nested[T]"]],
    Mapping[str, Union[T, "Nested[T]"]],
]
