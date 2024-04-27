import jax
from tqdm import tqdm
import optax
import pickle
import jaxlib
import jax.numpy as jnp
import matplotlib.pyplot as plt
from ml.dataset import Dataset

from jax import value_and_grad, jit
from ml.model import foward_deep_fm, foward_fm, foward_mlp, foward_embedding


@jit
def step(params: tuple, x: jaxlib.xla_extension.ArrayImpl, 
         y: jaxlib.xla_extension.ArrayImpl) -> tuple:
    """
    Make one update to a DeepFM

    Parameters:
    -----------
        params (list): a list of DeepFM parameters
        x (jaxlib.xla_extension.ArrayImpl): a minibatch of features
        y (jaxlib.xla_extension.ArrayImpl): a minibatch of targets

    Returns:
    --------
        A tuple of updated parameters, loss, gradients, and accuracy
    """

    def loss_fn(params, x, y):
        ys = foward_deep_fm(params, x)
        ys = jnp.clip(ys, 1e-7, 1 - 1e-7)

        return -jnp.mean(y * jnp.log(ys) + (1 - y) * jnp.log(1 - ys))

    loss, grads = value_and_grad(loss_fn)(params, x, y)
    predicted_class = jnp.round(foward_deep_fm(params, x))
    accuracy = jnp.mean(predicted_class == y)
    return params, loss, grads, accuracy


def train(params: list, data: Dataset, num_epochs: int):
    """
    Train a deep factorization machine, visualize the results, and save the weights

    Parameters:
    -----------
        params (list): a list of DeepFM parameters
        data (Dataset): a Dataset object
        num_epochs (int): the number of epochs to train for

    Returns:
    --------
        A tuple containing lists of epochs, loss, accuracy, and parameters
    """
    epochs, loss_list, acc_list = [], [], []
    solver = optax.adam(0.0001)
    solver_state = solver.init(params)

    for epoch in tqdm(range(num_epochs)):
        for x_batch, y_batch in data:
            params, loss, grads, acc = step(params, x_batch, y_batch)
            updates, solver_state = solver.update(grads, solver_state)
            params = optax.apply_updates(params, updates)

        epochs.append(epoch + 1)
        loss_list.append(loss)
        acc_list.append(acc)

        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Loss: {loss}, Accuracy: {acc}")

    plt.plot(epochs, loss_list, label='Loss')
    plt.plot(epochs, acc_list, label='Accuracy')
    plt.legend()
    plt.title("Training Loss and Accuracy")
    plt.savefig('ml/ml/figures/training_curves.jpg')

    # Saving the weights
    with open('ml/ml/weights/parameters.pickle', 'wb') as file:
        pickle.dump(params, file)

    return epochs, loss_list, acc_list, params


def predict(params: list, X: jax.Array) -> jaxlib.xla_extension.ArrayImpl:
    """
    Predict the probability of a user RSVPing to an event

    Parameters:
    -----------
        params (tuple): the parameters used to initialize the DeepFM being used to 
            make predictions
        X (array): an array of features to use for prediction

    Returns:
    --------
        predictions (array): predicted probabilities of users RSVPing for events
    """
    embedding_params, fm_params, mlp_params = params
    embeddings = foward_embedding(embedding_params, X)
    fm_out = foward_fm(fm_params, embeddings)
    mlp_out = foward_mlp(mlp_params, embeddings)
    y = jax.nn.sigmoid(fm_out + mlp_out)

    return y
