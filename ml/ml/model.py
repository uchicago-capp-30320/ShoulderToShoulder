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

    return embeddings


def init_fm(seed: int, embedding_params: jaxlib.xla_extension.ArrayImpl) -> list:
    """
    Initilaize weights to a factorization machine

    Parameters:
    -----------
        seed (int): a seed for the random number generator
        embeddings (Array): an embedding array

    Returns:
    --------
        params (list): a list that contains embeddings, the fc parameters, and the 
            linear layer parameters
    """
    params = []
    new_key = random.PRNGKey(seed)
    params.append(dict(embedding_weights=embedding_params[0]["embedding_weights"]))

    # The fully connected layer of the factorization machine
    # Dimensions: number of data points by one
    new_seed = int(random.randint(new_key, (1,), -jnp.inf, jnp.inf)[0])
    fully_conn = init_embedding_params(new_seed, 
                                       embedding_params[0]["embedding_weights"]
                                       .shape[0], 1
                                    )
    params.append(dict(embedding_weights=fully_conn[0]["embedding_weights"]))

    new_key, subkey = random.split(new_key)
    new_seed = int(random.randint(new_key, (1,), -jnp.inf, jnp.inf)[0])
    linear_layer = xavier(new_seed, 1, 1)
    params.append(dict(linear_weights=linear_layer, 
                       biases=jnp.zeros(linear_layer.shape[1])))

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
    X_emb = foward_embedding([params[0]], X)
    fc = jnp.sum(foward_embedding([params[1]], X), axis=1).reshape((-1, 1))
    linear = params[2]
    power_of_sums = (jnp.sum(X_emb, axis=1) ** 2).reshape((-1, 1))
    sums_of_powers = jnp.sum(X_emb ** 2, axis=1).reshape((-1, 1))
    linear_out = jnp.dot(fc, linear["linear_weights"]) + linear["biases"]
    raw_output = 0.5 * jnp.sum(power_of_sums - sums_of_powers, axis=1)

    return jnp.sum(linear_out + raw_output, axis=1)


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


def init_deep_fm(vocab_length: int, hidden_factors: int, layer_widths: list[int], 
                 seeds: tuple[int]=(1, 42, 99)) -> tuple:
    """
    Initialize parameters for a deep factorization machine

    Parameters:
    -----------
        vocab_length (int): the number of unique values across all features
        hidden_factors (int): the number of hidden factors to consider
        layer_widths (list[int]): dimensions for MLP layers, which should be 
        [num_input_features, desired_output, desired_output = new_input, 
        new_desired_output, ...]
        seeds (tuple[int]): a tuple of three integers to use for initialization

    Returns:
    --------
        params (tuple): parameters for the embeddings, MLP, and factorization model
    """
    seed_1, seed_2, seed_3 = seeds
    embedding_params = init_embedding_params(seed_1, vocab_length, hidden_factors)
    fm_params = init_fm(seed_2, embedding_params)
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
    _, fm_params, mlp_params = params
    fm_out = foward_fm(fm_params, X)
    mlp_out = foward_mlp(mlp_params, X)
    y = jax.nn.sigmoid(fm_out + mlp_out.T).T

    return y
