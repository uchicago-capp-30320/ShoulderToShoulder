import jax.numpy as jnp
import jaxlib
from jax.random import key, permutation, split

class Dataset:
    """
    A class to hold Jax arrays and generate random minibatches

    Parameters:
    -----------
        X (jaxlib.xla_extension.ArrayImpl): an aray of features from which to generate 
            minibatches
        Y (jaxlib.xla_extension.ArrayImpl): an array of targets from which to generate 
            minibatches
        batch_size (int): the number of data points in each batch
        seed (int): a seed for the random number generator to generate random minibatches
    """

    def __init__(self, X: jaxlib.xla_extension.ArrayImpl, Y: jaxlib.xla_extension.ArrayImpl, 
                 batch_size: int, seed: int) -> None:
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.key = key(seed)

        # Initial randomization
        self.shuffle()
        self.num_splits = self.X.shape[0]//batch_size + self.X.shape[0]%batch_size
        self.split()  # Split X and Y into lists of minibatches

    def shuffle(self) -> None:
        """Shuffle the arrays"""
        self.X = permutation(self.key, self.X, independent=True)
        self.Y = permutation(self.key, self.Y, independent=True)
        new_key, subkey = split(self.key)
        self.key = new_key

    def split(self):
        """Split the arrays into minibatches"""
        self.x_list = jnp.array_split(self.X, self.num_splits, axis=0)
        self.y_list = jnp.array_split(self.Y, self.num_splits, axis=0)

    def reset(self):
        """Generate new random minibatches"""
        self.shuffle()
        self.split()

    def __iter__(self):
        return self
    
    def __next__(self) -> tuple:

        # If the lists of x and y minibatches are empty, raise the StopIteration exceptioin
        if not self.x_list:
            raise StopIteration("Call the reset method to continue generating batches")

        x, y = self.x_list.pop(0), self.y_list.pop(0)

        return x, y
    
