# Taken from blackjax.types
from typing import Any, Iterable, Mapping, NamedTuple, TypeAlias, Union

import jax

Array: TypeAlias = jax.Array
ArrayLike: TypeAlias = jax.typing.ArrayLike

ArrayTree: TypeAlias = Union[Array, Iterable["ArrayTree"], Mapping[Any, "ArrayTree"]]
ArrayLikeTree: TypeAlias = Union[
    ArrayLike, Iterable["ArrayLikeTree"], Mapping[Any, "ArrayLikeTree"]
]

PRNGKey: TypeAlias = jax.Array
State: TypeAlias = NamedTuple
