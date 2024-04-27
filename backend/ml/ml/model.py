import jax
import jax.numpy as jnp
import jaxlib
from jax import random


def xavier(seed: int, n_in: int, n_out: int) -> jaxlib.xla_extension.ArrayImpl:
    """
    Xavier initialization from a normal distribution

    Params:
    -------
        seed (int): a seed for the random number generator
        n_in (int): number of inputs to the layer being initialized
        n_out (int): number of outputs from the layer being initialized

    Returns:
    --------
        Array: the intitialized array and new key
    """
    key = random.PRNGKey(seed)
    return jnp.sqrt(2/(n_in + n_out)) * random.normal(key, (n_in, n_out))


def init_embedding_params(seed: int, vocab_length: int, output_size: int) -> tuple:
    """
    Initialize an embedding layer

    Params:
    -------
        seed (int): an intial seed for the random number generator
        vocab_length (int): the number of unique input values
        output_size (int): the length of each vector

    Returns:
    --------
        tuple[Array, PRNGKey]: an aray of shape vocab length x output_size and new key
    """
    params = []
    new_key, subkey = random.PRNGKey(seed)
    initial_weights = xavier(subkey, vocab_length, output_size)
    params.append(dict(embedding_weights=initial_weights))

    return params


def foward_embedding(params: list[dict], X: jax.Array) -> jaxlib.xla_extension.ArrayImpl:
    """
    Get embeddings for each feature

    Parameters:
    -----------
        params (list[dict]): a list with a dictionary containing initialized weights
        X (Array): a jax array to get embeddings for

    Returns:
    --------
        Array: a jax array of embeddings for each data point where each embedding is
        the number of features by the embedding dimension
    """
    embedding_weights = params[0]["embedding_weights"]
    embeddings = jnp.take(embedding_weights, jnp.astype(X, int), axis=0)
    embeddings = embeddings.reshape(embeddings.shape[0],  # Concatenate embeddings into rows
                                    embeddings.shape[1] * embeddings.shape[2])

    return embeddings


def init_fm(seed: int, num_features: int, num_factors: int) -> list:
    """
    Initilaize weights to a factorization machine

    Parameters:
    -----------
        seed (int): a seed for the random number generator
        num_features: the number of input features
        num_factors: the number of factors to consider

    Returns:
    --------
        params (list): a list that contains dictionaries of the w, V, and bias terms for a 
            factorization machine
    """
    params = []
    new_key = random.PRNGKey(seed)

    w = xavier(seed, num_features, 1)
    new_seed = int(random.randint(new_key, (1,), -jnp.inf, jnp.inf)[0])
    V = xavier(new_seed, num_features, num_factors)
    bias = 0.0
    
    params.extend([dict(w=w), dict(V=V), dict(bias=bias)])

    return params


def foward_fm(params: list, X: jax.Array) -> jaxlib.xla_extension.ArrayImpl:
    """
    Calculate one foward pass of a factorization machine

    Parameters:
    -----------
        params (list): a list of parameters for the factorization machine
        X (Array): a jax numpy aray

    Returns:
    --------
        scores (Array): an array of raw scores from the factorization machine
    """
    w, v, b = params
    linear_term = jnp.dot(X, w["w"]) + b["bias"]
    squares_of_sums = jnp.dot(X, v["V"])**2
    sums_of_squares = jnp.dot(X**2, v["V"]**2)

    # Calculate pairwise interactions
    interactions = 0.5 * (squares_of_sums - sums_of_squares).sum(axis=1)

    return (linear_term.T + interactions).T


def init_mlp_params(seed: int, layer_widths: list[int]) -> list:
    """
    Initialize a multilayer perceptron

    Parameters:
    -----------
        seed (int): a seed for the random number generator
        layer_widths (list[int]): a list of integers for input and output dimensions

    Returns:
    --------
        params (list): parameters for a multilayer perceptron
    """
    params = []
    key = random.PRNGKey(seed)

    for n_in, n_out in zip(layer_widths[:-1], layer_widths[1:]):
        weight_key, _ = jax.random.split(key)
        new_seed = int(random.randint(weight_key, (1,), -jnp.inf, jnp.inf)[0])
        params.append(
            dict(weights=xavier(new_seed, n_in, n_out),
                 biases=jnp.zeros(shape=n_out))
            )

    return params


def foward_mlp(params: list, X: jax.Array, dropout: int=0.01, 
               train: bool=True) -> jaxlib.xla_extension.ArrayImpl:
    """
    Execute one foward pass for a multilayer perceptron

    Parameters:
    -----------
        params (list): a list of model parameters
        X (Array): features to use for prediction
        dropout (float): the proportion of neurons to zero out during training
        train (bool): whether this function is being called for training or inference

    Returns:
    --------
        y (Array): predicted values
    """
    x = jnp.dot(X, params[0]["weights"]) + params[0]["biases"]
    x = jax.nn.relu(x)

    # We shouldn't apply dropout for predictions
    if train:
        mask = jax.random.uniform(random.key(x[0, 0].astype(int)), x.shape) > dropout
        x = jnp.asarray(mask, dtype=jnp.float32) * x / (1.0 - dropout)

    for layer in params[1:]:
        x = jnp.dot(x, layer["weights"]) + layer["biases"]
        x = jax.nn.relu(x)

        if train:
            mask = jax.random.uniform(random.key(x[0, 0].astype(int)), x.shape) > dropout
            x = jnp.asarray(mask, dtype=jnp.float32) * x / (1.0 - dropout)

    return x


def init_deep_fm(vocab_length: int, num_features: int, num_factors: int, 
                 seeds: tuple[int]=(1, 42, 99)) -> tuple:
    """
    Initialize parameters for a deep factorization machine

    Parameters:
    -----------
        vocab_length (int): the number of unique values across all features
        num_features (int): the number of features in the dataset
        num_factors (int): the number of hidden factors to consider
        seeds (tuple[int]): a tuple of three integers to use for initialization

    Returns:
    --------
        params (tuple): parameters for the embeddings, MLP, and factorization model
    """
    seed_1, seed_2, seed_3 = seeds
    embedding_params = init_embedding_params(seed_1, vocab_length, num_factors)
    n_in = embedding_params[0]['embedding_weights'].shape[1]
    layer_widths = [n_in * num_features, 128, 128, 64, 64, 32, 32, 1]
    fm_params = init_fm(seed_2, num_features * n_in, num_factors)
    mlp_params = init_mlp_params(seed_3, layer_widths)

    return embedding_params, fm_params, mlp_params


def foward_deep_fm(params: list, X: jax.Array) -> jaxlib.xla_extension.ArrayImpl:
    """
    Calculate the foward pass of a deep factorization machine

    Parameters:
        params (list): a list of parameters for computing a foward pass
        X (jax.Array): features to use in the foward pass

    Returns:
    --------
        predictions (jax.Array): predicted probabilities of users RSVPing to events
    """
    embedding_params, fm_params, mlp_params = params
    embeddings = foward_embedding(embedding_params, X)
    fm_out = foward_fm(fm_params, embeddings)
    mlp_out = foward_mlp(mlp_params, embeddings)
    y = jax.nn.sigmoid(fm_out + mlp_out.T).T

    return y
