import os
import jax
import optax
import pickle
import jaxlib
import requests
from tqdm import tqdm
import jax.numpy as jnp
import matplotlib.pyplot as plt
from shoulder.ml.ml.dataset import Dataset

from jax import value_and_grad, jit
from shoulder.ml.ml.model import foward_deep_fm, foward_fm, foward_mlp, foward_embedding


def preprocess(raw_data: requests.models.Response) -> jaxlib.xla_extension.ArrayImpl:
    """
    Prepare data for training or predicting

    Parameters:
    -----------
        raw_data (requests.models.Response): a response from the event suggestions database

    Returns:
        a tuple of preprocessed arrays for training or predicting
    """
    feature_list, target_list = [], []
    raw_json = raw_data.json()
    json_results = raw_json["results"]

    # json_results is a list of dictionaries
    for d in json_results:
        del d["id"]
        user_id = d["user_id"]  # We will add this after everything else
        del d["user_id"]

        if d["attended_event"] == 1:
            target_list.append(1)
        else:
            target_list.append(0)
        del d["attended_event"]

        user_event_list = []
        low, high = 0, 1

        for key in sorted(d.keys()):
            if d[key] is True:
                user_event_list.append(high)
            else:
                user_event_list.append(low)

            # Ensures unique integers for every response in very field, basically creating
            # a vocabulary for the embedding layer
            low += 2
            high += 2

        # Makes sure the user ID doesn't overlap with another "token"
        user_event_list.append(high + user_id)

        feature_list.append(user_event_list)

    x, y= jnp.array(feature_list, dtype=float), jnp.array(target_list, dtype=float)

    return x, y


def save_outputs(epochs: list, loss_list: list, acc_list: list, params: list) -> None:
    """
    Save diagnostic plots and weights from training a DeepFM

    Parameters:
    -----------
        epochs (list): a list of training epochs
        loss_list (list): a list of losses at each epoch
        acc_list (list): a list of accuracy for each epoch
        params (list): a list of model parameters
    """
    # Make sure we remove old plots
    if os.path.isfile('ml/ml/figures/training_curves.jpg'):
        os.remove('ml/ml/figures/training_curves.jpg')

    plt.plot(epochs, loss_list, label='Loss')
    plt.plot(epochs, acc_list, label='Accuracy')
    plt.legend()
    plt.title("Training Loss and Accuracy")
    plt.savefig('figures/training_curves.jpg')
    plt.close()

    # Saving the weights
    with open('weights/parameters.pkl', 'wb') as file:
        pickle.dump(params, file)


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

    save_outputs(epochs, loss_list, acc_list, params)

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
    # Check if params is a global variable and if not, read them from a pkl file and add
    # the parameters to the gloabl scoe so we don't have to keep reading them in when we
    # call predict
    if "params" in globals():
        params = globals["params"]
    else:
        with open('ml/ml/weights/parameters.pkl', 'rb') as file:
            params = pickle.load(file)

        globals()["params"] = params

    embedding_params, fm_params, mlp_params = params
    embeddings = foward_embedding(embedding_params, X)
    fm_out = foward_fm(fm_params, embeddings)
    mlp_out = foward_mlp(mlp_params, embeddings)
    y = jax.nn.sigmoid(fm_out + mlp_out)

    return y
