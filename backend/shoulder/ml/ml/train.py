import os
import jax
import optax
import pickle
import jaxlib
from tqdm import tqdm
import jax.numpy as jnp
import matplotlib.pyplot as plt
from shoulder.ml.ml.dataset import Dataset
import pathlib

from jax import value_and_grad, jit
from shoulder.ml.ml.model import foward_deep_fm, foward_fm, foward_mlp, foward_embedding

LR = 0.0001
WEIGHTS = None
TRAINING_CURVES_PATH = os.path.join(pathlib.Path(__file__).parent, "figures/training_curves.jpg")
PARAMETERS_PATH = os.path.join(pathlib.Path(__file__).parent, "weights/parameters.pkl")


def _ensure_weights():
    """Add the pretrained weights to the global scope"""
    global WEIGHTS
    if WEIGHTS is None:
        with open(PARAMETERS_PATH, 'rb') as file:
            WEIGHTS = pickle.load(file)


def save_outputs(epochs: list, loss_list: list, acc_list: list, params: list,
                 path: str=PARAMETERS_PATH) -> None:
    """
    Save diagnostic plots and weights from training a DeepFM

    Parameters:
    -----------
        epochs (list): a list of training epochs
        loss_list (list): a list of losses at each epoch
        acc_list (list): a list of accuracy for each epoch
        params (list): a list of model parameters
        path (str): a path for saving the weights
    """    
    # Saving the weights
    with open(path, 'wb') as file:
        pickle.dump(params, file)

    plot_training_curves(epochs, loss_list, acc_list)


def plot_training_curves(epochs: list, loss_list: list, acc_list: list) -> None:
    """
    Plot training curves and save the plot to a file

    Parameters:
    -----------
        epochs (list): a list of training epochs
        loss_list (list): a list of losses at each epoch
        acc_list (list): a list of accuracy for each epoch
    """
    if os.path.isfile(TRAINING_CURVES_PATH):
        os.remove(TRAINING_CURVES_PATH)

    plt.figure()
    plt.plot(epochs, loss_list, label='Loss')
    plt.plot(epochs, acc_list, label='Accuracy')
    plt.legend()
    plt.title("Training Loss and Accuracy")
    plt.savefig(TRAINING_CURVES_PATH)
    plt.close()

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


def train(params: list, data: Dataset, num_epochs: int,
          path: str=PARAMETERS_PATH):
    """
    Train a deep factorization machine, visualize the results, and save the weights

    Parameters:
    -----------
        params (list): a list of DeepFM parameters
        data (Dataset): a Dataset object
        num_epochs (int): the number of epochs to train for
        path (str): a location to write the weights to

    Returns:
    --------
        A tuple containing lists of epochs, loss, accuracy, and parameters
    """
    epochs, loss_list, acc_list = [], [], []
    solver = optax.adam(LR)
    solver_state = solver.init(params)

    for epoch in tqdm(range(num_epochs)):
        for x_batch, y_batch in data:
            params, loss, grads, acc = step(params, x_batch, y_batch)
            updates, solver_state = solver.update(grads, solver_state)
            params = optax.apply_updates(params, updates)

        epochs.append(epoch + 1)
        loss_list.append(float(loss))  # Ensure loss is a float
        acc_list.append(float(acc))    # Ensure accuracy is a float

        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Loss: {loss}, Accuracy: {acc}")

    save_outputs(epochs, loss_list, acc_list, params, path)
    return epochs, loss_list, acc_list, params


def predict(X: jax.Array) -> jaxlib.xla_extension.ArrayImpl:
    """
    Predict the probability of a user RSVPing to an event

    Parameters:
    -----------
        X (array): an array of features to use for prediction

    Returns:
    --------
        predictions (array): predicted probabilities of users RSVPing for events
    """
    _ensure_weights()
    params = WEIGHTS

    embedding_params, fm_params, mlp_params = params
    embeddings = foward_embedding(embedding_params, X)

    # Makes zero vectors for user IDs not in the training data
    embeddings = jnp.nan_to_num(embeddings)

    fm_out = foward_fm(fm_params, embeddings)
    mlp_out = foward_mlp(mlp_params, embeddings, train=False)
    y = jax.nn.sigmoid(fm_out + mlp_out)

    return y
