import jax.numpy as jnp
import math
from jax.random import key, permutation, split

class Dataset:
    """
    A class to hold Jax arrays and generate random minibatches

    Parameters:
    -----------
        X (jaxlib.xla_extension.ArrayImpl): an array of features from which to generate minibatches
        Y (jaxlib.xla_extension.ArrayImpl): an array of targets from which to generate minibatches
        batch_size (int): the number of data points in each batch
        seed (int): a seed for the random number generator to generate random minibatches
    """

    def __init__(self, X: jnp.ndarray, Y: jnp.ndarray, batch_size: int, seed: int) -> None:
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.batch_index = jnp.inf
        self.key = key(seed)

        self.num_samples = X.shape[0]
        self.num_batches = math.ceil(self.num_samples / batch_size)
        self._shuffle()

    def _shuffle(self):
        """Reshuffle the dataset"""
        perm = permutation(self.key, jnp.arange(self.num_samples))
        self.shuffled_X = self.X[perm]
        self.shuffled_Y = self.Y[perm]

        self.x_batches = jnp.array_split(self.shuffled_X, self.num_batches)
        self.y_batches = jnp.array_split(self.shuffled_Y, self.num_batches)

        self.batch_index = 0

        self.key, subkey = split(self.key)

    def __iter__(self):
        return self

    def __next__(self) -> tuple:
        """Generate next minibatch"""

        # Reshuffles the dataset once the end has been reached
        if self.batch_index >= self.num_batches:
            self._shuffle()
            raise StopIteration

        x_batch = self.x_batches[self.batch_index]
        y_batch = self.y_batches[self.batch_index]
        self.batch_index += 1

        return x_batch, y_batch
