"""Type definitions."""

from typing import Any, Callable, Mapping, Sequence, TypeAlias, TypeVar, Union

import keras

"""
A tensor in any of the backends.

We do not define it explicitly to not require all the backends to be installed
and imported. The explicit definition would be:
```
numpy.ndarray,
| tensorflow.Tensor,
| tensorflow.RaggedTensor,
| tensorflow.SparseTensor,
| tensorflow.IndexedSlices,
| jax.Array,
| jax.experimental.sparse.JAXSparse,
| torch.Tensor,
| keras.KerasTensor,
```
"""
Tensor: TypeAlias = Any

Shape: TypeAlias = Sequence[int | None]

DType: TypeAlias = str

ConstraintLike: TypeAlias = (
    str
    | keras.constraints.Constraint
    | type[keras.constraints.Constraint]
    | Callable[[Tensor], Tensor]
)

InitializerLike: TypeAlias = (
    str
    | keras.initializers.Initializer
    | type[keras.initializers.Initializer]
    | Callable[[Shape, DType], Tensor]
    | Tensor
)

RegularizerLike: TypeAlias = (
    str
    | keras.regularizers.Regularizer
    | type[keras.regularizers.Regularizer]
    | Callable[[Tensor], Tensor]
)

T = TypeVar("T")
Nested: TypeAlias = (
    T | Sequence[Union[T, "Nested[T]"]] | Mapping[str, Union[T, "Nested[T]"]]
)
