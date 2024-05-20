import jaxlib
import jax.numpy as jnp
from shoulder.ml.ml.dataset import Dataset


def test_dataset_init():
    x = jnp.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]])
    y = jnp.array([1, 2, 3, 4, 5, 6])
    test_data = Dataset(x, y, 2, 9)
    assert not jnp.array_equal(test_data.shuffled_X, x), "Check whether the arrays are permuted"  # The arrays should have been permuted
    assert not jnp.array_equal(test_data.shuffled_Y, y), "Check whether the arrays are permuted"

    assert len(test_data.x_batches) == 3, "Ensure correct number of x_batches"
    assert len(test_data.y_batches) == 3, "Ensure correct number of y_batches"


def test_dataset_iteration():
    x = jnp.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]])
    y = jnp.array([1, 2, 3, 4, 5, 6])
    test_data = Dataset(x, y, 2, 9)

    for x_batch, y_batch in test_data:
        assert type(x_batch) == jaxlib.xla_extension.ArrayImpl, "Ensure x_batch is a jax array"
        assert type(y_batch) == jaxlib.xla_extension.ArrayImpl, "Ensure y_batch is a jax array"

