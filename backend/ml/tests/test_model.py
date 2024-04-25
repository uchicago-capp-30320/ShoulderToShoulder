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
    assert type(foward_embedding(test_embedding_params, 
                                 emb)) == jaxlib.xla_extension.ArrayImpl


def test_init_fm():
    test_embedding_params = init_embedding_params(seed=17, vocab_length=10, output_size=3)
    test_init_fm = init_fm(90, test_embedding_params)
    assert jax.tree.map(lambda x: x.shape, 
                        test_init_fm) == [{'embedding_weights': (10, 3)},
                                          {'embedding_weights': (10, 1)},
                                          {'biases': (1,), 'linear_weights': (1, 1)}]
    

def test_foward_fm():
    test_embedding_params = init_embedding_params(seed=17, vocab_length=10, output_size=3)
    test_init_fm = init_fm(90, test_embedding_params)
    emb = jnp.array([[1, 2, 3, 8], [0, 5, 9, 7], [8, 7, 1, 0]])
    assert type(foward_fm(test_init_fm, emb)) == jaxlib.xla_extension.ArrayImpl


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


def test_init_deep_fm():
    test_deep_fm_params = init_deep_fm(500, 10, [50, 20, 20, 1])
    tree = jax.tree.map(lambda x: x.shape, test_deep_fm_params)
    out = ([{'embedding_weights': (500, 10)}],[{'embedding_weights': (500, 10)},
                                               {'embedding_weights': (500, 1)},
                                               {'biases': (1,), 'linear_weights': (1, 1)}],
            [{'biases': (20,), 'weights': (50, 20)},{'biases': (20,), 'weights': (20, 20)},
             {'biases': (1,), 'weights': (20, 1)}])
    assert tree == out


def test_foward_deep_fm():
    x_train = jnp.astype(random.randint(random.PRNGKey(59), shape=(1000, 5), minval=0, 
                                        maxval=50), float)
    train_params = init_deep_fm(51, 5, [5, 64, 64, 32, 32, 1])
    out = foward_deep_fm(train_params, x_train)
    assert type(out) == jaxlib.xla_extension.ArrayImpl