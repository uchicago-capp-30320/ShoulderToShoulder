import jax
from tqdm import tqdm
import optax
import jaxlib
import jax.numpy as jnp

from jax import value_and_grad, jit
from ml.model import foward_deep_fm, foward_fm, foward_mlp


@jit
def step(params: list, x: jax.Array, y: jax.Array) -> jaxlib.xla_extension.ArrayImpl:
    """
    Update the parameters of a deep factorization machine

    Parameters:
    -----------
        params (list): parameters for the model
        x (Array): a jax numpy array
        y (Array): a jax numpy array

    Returns:
    --------
    
    """

    # Binary cross entropy with clipping to avoid rounding issues
    def loss_fn(params, x, y):
        ys = foward_deep_fm(params, x)
        ys = jnp.clip(ys, 1e-7, 1 - 1e-7)

        return -jnp.mean(y * jnp.log(ys) + (1 - y) * jnp.log(1 - ys))

    predicted_class = jnp.round(foward_deep_fm(params, x))
    accuracy = jnp.mean(predicted_class == y)

    # grads also returns a pytree
    loss, grads = value_and_grad(loss_fn)(params, x, y)

    # Return the state and update
    return params, loss, grads, accuracy


def train(params: list, X: jax.Array, Y: jax.Array, 
          num_epochs: int) -> jaxlib.xla_extension.ArrayImpl:
    """
    Train a deep factorization machine

    Parameters:
    -----------
        params (list): parameters for the model
        X (Array): features to use for predicting user interactions
        Y (Array): user interactions
        num_epochs (int): the number of epochs to train for

    Returns:
    --------
        tuple (list): a list of epochs, a list of the loss at each epoch, and a list 
            of the accuracy at each epoch
    """
    epochs, loss_list, acc_list = [], [], []

    solver = optax.adam(0.0001)
    solver_state = solver.init(params)

    # Main training loop
    for epoch in tqdm(range(num_epochs)):
        params, loss, grads, acc = step(params, X, Y)

        # Update the weights
        updates, solver_state = solver.update(grads, solver_state, params)
        params = optax.apply_updates(params, updates)

        epochs.append(epoch)
        loss_list.append(loss)
        acc_list.append(acc)

        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Loss: {loss}, Accuracy: {acc}")

    return epochs, loss_list, acc_list, params


def predict(params: list, X: jax.Array) -> jaxlib.xla_extension.ArrayImpl:
    """
    Predict the probability of a user RSVPing to an event

    Parameters:
    -----------
        params (tuple): the parameters used to initialize the DeepFM being used to 
            make predictions
        X (array): ana rray of features to use for prediction

    Returns:
    --------
        predictions (array): predicted probabilities of users RSVPing for events
    """
    _, fm_params, mlp_params = params
    mlp_out = foward_mlp(mlp_params, X[0], train=False)
    fm_out = foward_fm(fm_params, X[0])
    y = jax.nn.sigmoid(fm_out + mlp_out.T).T

    return y
