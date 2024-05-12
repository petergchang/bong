from functools import partial
from typing import Callable, Tuple

import numpy as np
import jax
from jax import vmap
from jax.lax import scan
import jax.numpy as jnp
import jax.random as jr
from jax.tree_util import tree_map
import tensorflow_datasets as tfds


def _process(
    dataset: Tuple,
    n: int=None,
    key: int=0,
    shuffle: bool=True,
    oh: bool=True,
    output_dim: int=10,
) -> Tuple:
    """Process a single element.
    """
    X, *args, Y = dataset
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    Y = jax.nn.one_hot(Y, output_dim) if oh else Y
    idx = jr.permutation(key, jnp.arange(len(X))) if shuffle \
        else jnp.arange(len(X))
    X, Y = X[idx], Y[idx]
    if n is not None:
        X, Y = X[:n], Y[:n]
    new_args = []
    for arg in args:
        if isinstance(arg, dict):
            arg = tree_map(lambda x: x[idx], arg)
        else:
            arg = arg[idx]
        new_args.append(arg)
    return X, *new_args, Y


def process_dataset(
    train: Tuple,
    val: Tuple,
    test: Tuple,
    ntrain: int=None,
    nval: int=None,
    ntest: int=None,
    key: int=0,
    shuffle: bool=True,
    oh_train: bool=True,
    output_dim: int=10,
) -> dict:
    """Wrap dataset into a dictionary.
    """
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    keys = jr.split(key, 3)
    train, val, test = \
        (_process(dataset, n, key, shuffle, oh, output_dim)
         for dataset, n, key, oh in zip([train, val, test], 
                                        [ntrain, nval, ntest],
                                        keys, [oh_train, False, False]))
    dataset = {
        'train': train,
        'val': val,
        'test': test,
    }
    return dataset


def load_base_dataset(
    data_dir: str="/tmp/data",
    dataset_type: str='fashion_mnist',
    **process_kwargs,
) -> dict:
    """Load train, validatoin, and test datasets into memory.
    """
    ds_builder = tfds.builder(dataset_type, data_dir=data_dir)
    ds_builder.download_and_prepare()
    
    train_ds, val_ds, test_ds = \
        (tfds.as_numpy(ds_builder.as_dataset(split=split, batch_size=-1)) 
         for split in ['train[10%:]', 'train[:10%]', 'test'])
    
    # Normalize pixel values
    for ds in [train_ds, val_ds, test_ds]:
        ds['image'] = np.float32(ds['image']) / 255.
    
    output_dim = train_ds["label"].max() + 1
    train, val, test = \
        ((jnp.array(ds['image']), jnp.array(ds['label'])) 
         for ds in [train_ds, val_ds, test_ds])
    
    dataset = process_dataset(train, val, test, output_dim=output_dim,
                              **process_kwargs)
    return dataset


def generate_stationary_experiment(
    ntrain: int,
    nval: int=500,
    ntest: int=1_000,
):
    kwargs = {
        "ntrain": ntrain,
        "nval": nval,
        "ntest": ntest,
    }
    dataset = {
        "load_fn": partial(load_base_dataset, **kwargs),
        "configs": kwargs,
    }
    return dataset