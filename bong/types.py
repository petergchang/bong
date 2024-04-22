# Taken from blackjax.types
from typing import Any, Iterable, Mapping, Union

import jax

Array = jax.Array
ArrayLike = jax.typing.ArrayLike

ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[Any, "ArrayTree"]]
ArrayLikeTree = Union[
    ArrayLike, Iterable["ArrayLikeTree"], Mapping[Any, "ArrayLikeTree"]
]

PRNGKey = jax.Array