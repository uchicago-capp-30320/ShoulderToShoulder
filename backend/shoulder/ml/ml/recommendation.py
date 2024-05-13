import pickle
import jaxlib
import requests
import jax.numpy as jnp
from shoulder.ml.ml.dataset import Dataset
from shoulder.ml.ml.model import init_deep_fm
from shoulder.ml.ml.train import preprocess, train, predict


def pretrain(raw_data: requests.models.Response, num_factors: int=5, batch_size=32, 
             num_epochs: int=10, seed=1994, seeds=(8, 6, 7)) -> tuple[list]:
    """
    Pretrain a DeepFM

    Parameters:
    -----------
        raw_data (requests.models.Response): a response from the event suggestions table.
        num_factors (int): the number of latent factors to consider for the FM.
        batch_size (int): the number of training examples in each batch.
        num_epochs (int): the number of passes to perform on the dataset durind training.
        seed (int): a seed for shuffling the data
        seeds (tuple): a tuple of three seeds for model initialization.

    Returns:
    --------
        A tuple of lists of epochs, loss, and accuracy
    """
    full_x, full_y = preprocess(raw_data)
    data = Dataset(full_x, full_y, batch_size, seed)
    params = init_deep_fm(int(jnp.max(full_x)), full_x.shape[1], num_factors, seeds)
    epochs, loss_list, acc_list, params = train(params, data, num_epochs)
    return epochs, loss_list, acc_list


def finetune(raw_data: requests.models.Response,  batch_size=32, num_epochs: int=5, 
             seed=1999) -> tuple[list]:
    """
    Finetune a DeepFM

    Parameters:
    -----------
        raw_data (requests.models.Response): a response from the event suggestions table.
        batch_size (int): the number of training examples in each batch.
        num_epochs (int): the number of passes to perform on the dataset durind training.
        seed (int): a seed for shuffling the data

    Returns:
    --------
        A tuple of lists of epochs, loss, and accuracy
    """
    full_x, full_y = preprocess(raw_data)
    data = Dataset(full_x, full_y, batch_size, seed)

    with open('weights/parameters.pkl', 'rb') as file:
            params = pickle.load(file)

    epochs, loss_list, acc_list, params = train(params, data, num_epochs)

    return epochs, loss_list, acc_list


def recommend(raw_data: requests.models.Response) -> jaxlib.xla_extension.ArrayImpl:
     """
     Get recommendations for users and events.

     Parameters:
     -----------
        raw_data (requests.models.Response): user and event data from the event suggestions 
            table.

        Returns:
        --------
            A jax NumPy array of predicted probabilities of attending events
     """
     full_x = preprocess(raw_data)
     return predict(full_x)
