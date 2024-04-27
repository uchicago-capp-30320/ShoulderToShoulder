from ml.dataset import Dataset
import jaxlib
import jax.numpy as jnp


def test_dataset_init():
    x = jnp.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]])
    y = jnp.array([1, 2, 3, 4, 5, 6])
    test_data = Dataset(x, y, 2, 9)
    assert not jnp.array_equal(test_data.shuffled_X, x)  # The arrays should have been permuted
    assert not jnp.array_equal(test_data.shuffled_Y, y)

    assert len(test_data.x_batches) == 3
    assert len(test_data.y_batches) == 3


def test_dataset_iteration():
    x = jnp.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]])
    y = jnp.array([1, 2, 3, 4, 5, 6])
    test_data = Dataset(x, y, 2, 9)

    for x_batch, y_batch in test_data:
        assert type(x_batch) == jaxlib.xla_extension.ArrayImpl
        assert type(y_batch) == jaxlib.xla_extension.ArrayImpl

    # Should be able to reset and continue getting new minibatches
    test_data.reset()
    assert type(next(test_data)) == tuple