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
        self.key = key(seed)

        self.num_samples = X.shape[0]
        self.num_batches = math.ceil(self.num_samples / batch_size)
        self.reset()

    def reset(self) -> None:
        """Shuffle the data and split into minibatches"""
        self.key, subkey = split(self.key)
        perm = permutation(subkey, jnp.arange(self.num_samples))
        self.shuffled_X = self.X[perm]
        self.shuffled_Y = self.Y[perm]

        self.x_batches = jnp.array_split(self.shuffled_X, self.num_batches)
        self.y_batches = jnp.array_split(self.shuffled_Y, self.num_batches)

        self.batch_index = 0

    def __iter__(self):
        return self

    def __next__(self) -> tuple:
        """Generate next minibatch"""
        if self.batch_index >= self.num_batches:
            self.reset()
            raise StopIteration

        x_batch = self.x_batches[self.batch_index]
        y_batch = self.y_batches[self.batch_index]
        self.batch_index += 1

        return x_batch, y_batch
    
