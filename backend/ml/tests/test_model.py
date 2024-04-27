from ml.model import xavier, init_embedding_params, init_fm, foward_embedding, foward_fm
from ml.model import init_mlp_params, foward_mlp, init_deep_fm, foward_deep_fm
import jax
from jax import random
import jaxlib
import jax.numpy as jnp


def test_xavier_initialization():
    assert jax.tree.map(lambda x: x.shape, xavier(22, 5, 5)) == (5, 5)


def test_init_embedding_params():
    assert jax.tree.map(lambda x: x.shape, 
                        init_embedding_params(22, 10, 5)) == [
                            {'embedding_weights': (10, 5)}
                            ]
    

def test_foward_embedding():
    emb = jnp.array([[1, 2, 3, 8], [0, 5, 9, 7], [8, 7, 1, 0]])
    test_embedding_params = init_embedding_params(seed=17, vocab_length=10, output_size=3)
    output = foward_embedding(test_embedding_params, emb)
    assert type(output) == jaxlib.xla_extension.ArrayImpl
    assert output.shape == (3, 12)


def test_init_fm():
    w, v, b = init_fm(90, 3, 10)
    assert w["w"].shape == (3, 1)
    assert v["V"].shape == (3, 10)
    assert b["bias"] == 0.0
    

def test_foward_fm():
    key = random.PRNGKey(9)
    x = random.normal(key, (10, 3))
    test_init_fm = init_fm(90, 3, 10)
    output = foward_fm(test_init_fm, x)
    assert type(output) == jaxlib.xla_extension.ArrayImpl
    assert output.shape == (10, 1)


def test_init_mlp_params():
    test_mlp_init = init_mlp_params(7, [2, 5, 5, 1])
    assert jax.tree.map(lambda x: x.shape, test_mlp_init) == [{'biases': (5,), 
                                                               'weights': (2, 5)},
                                                              {'biases': (5,), 
                                                                'weights': (5, 5)},
                                                              {'biases': (1,), 
                                                               'weights': (5, 1)}]
    

def test_foward_mlp():
    test_mlp_init = init_mlp_params(7, [2, 5, 5, 1])
    foward = foward_mlp(test_mlp_init, random.uniform(random.PRNGKey(129), shape=(10, 2)))
    assert type(foward) == jaxlib.xla_extension.ArrayImpl
    assert foward.shape == (10, 1)


def test_init_deep_fm():
    e, f, m = init_deep_fm(500, 10, 5)
    w, v, b = f
    assert jax.tree.map(lambda x: x.shape, e) == [{'embedding_weights': (500, 5)}]
    assert w["w"].shape == (50, 1)
    assert v["V"].shape == (50, 5)
    assert b["bias"] == 0.0
    assert jax.tree.map(lambda x: x.shape, m) == [{'biases': (128,), 'weights': (50, 128)},
                                                  {'biases': (128,), 'weights': (128, 128)},
                                                  {'biases': (64,), 'weights': (128, 64)},
                                                  {'biases': (64,), 'weights': (64, 64)}, 
                                                  {'biases': (32,), 'weights': (64, 32)}, 
                                                  {'biases': (32,), 'weights': (32, 32)}, 
                                                  {'biases': (1,), 'weights': (32, 1)}]


def test_foward_deep_fm():
    x_train = jnp.astype(random.randint(random.PRNGKey(59), shape=(1000, 5), minval=0, 
                                        maxval=50), float)
    train_params = init_deep_fm(500, 5, 5)
    out = foward_deep_fm(train_params, x_train)
    assert type(out) == jaxlib.xla_extension.ArrayImpl