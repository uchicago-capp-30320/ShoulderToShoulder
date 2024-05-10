import pickle
import jaxlib
import requests
import jax.numpy as jnp
from shoulder.ml.ml.dataset import Dataset
from shoulder.ml.ml.model import init_deep_fm
from shoulder.ml.ml.train import preprocess, train, predict

def preprocess_predict(json_results) -> jaxlib.xla_extension.ArrayImpl:
    """
    Prepare data for training or predicting

    Parameters:
    -----------
        raw_data (requests.models.Response): a response from the event suggestions database

    Returns:
        a tuple of preprocessed arrays for training or predicting
    """
    feature_list, target_list, event_id_list, user_id_list =[],[],[],[]

    # json_results is a list of dictionaries
    for d in json_results:
        del d["id"]
        user_id = d["user_id"]  # We will add this after everything else
        del d["user_id"]

        event_id = d['event_id']
        del d['event_id']

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

        user_id_list.append(user_id)
        event_id_list.append(event_id)

    x, y= jnp.array(feature_list, dtype=float), jnp.array(target_list, dtype=float)

    return x, y, user_id_list, event_id_list

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


def finetune(raw_data: requests.models.Response,  batch_size=32, num_epochs: int=101, 
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
     full_x, _, user_id_list, event_id_list = preprocess_predict(raw_data)
     return predict(full_x), user_id_list, event_id_list
