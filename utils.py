"""Simulate online stochastic gradient descent learning of a simple task."""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from collections.abc import Generator
from jax import Array

import jax
import jax.numpy as jnp

import equinox as eqx
import optax


def accuracy(pred_y: Array, y: Array) -> Array:
    """Compute elementwise accuracy."""
    predicted_class = jnp.where(pred_y > 0.5, 1., 0.) if check_pred_y_1d(pred_y.shape) else jnp.argmax(pred_y, axis=-1)
    return predicted_class == y

def mse(pred_y: Array, y: Array) -> Array:
    """Compute elementwise mean squared error."""
    return jnp.square(pred_y - y).sum(axis=-1)

def ce(pred_y: Array, y: Array) -> Array:
    """Compute elementwise cross-entropy loss."""
    pred_y = jnp.exp(pred_y) / jnp.sum(jnp.exp(pred_y), axis=-1, keepdims=True)
    y = jax.nn.one_hot(y, pred_y.shape[-1])
    return -jnp.sum(y * jnp.log(pred_y), axis=-1)

def batcher(sampler: Sequence, batch_size: int) -> Generator[Sequence, None, None]:
    """Batch a sequence of examples."""
    n = len(sampler)
    for i in range(0, n, batch_size):
        yield sampler[i : min(i + batch_size, n)]

@eqx.filter_value_and_grad
def compute_loss(model: eqx.Module, x: Array, y: Array, key: Array, loss_fn: Callable) -> Array:
    """Compute cross-entropy loss on a single example."""
    keys = jax.random.split(key, x.shape[0])
    pred_y = jax.vmap(model)(x, key=keys)
    loss = loss_fn(pred_y, y)
    return loss.mean()
