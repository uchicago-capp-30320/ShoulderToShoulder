from ml.train import step, train, predict
from jax import random
import jaxlib
from ml.model import init_deep_fm


def test_step():
    x_train = random.randint(random.PRNGKey(59), shape=(1000, 5), minval=0, 
                             maxval=50).astype(float)
    y_train = random.bernoulli(random.PRNGKey(997), 0.35, shape=(1000,)).astype(float)
    train_params = init_deep_fm(51, 5, [5, 64, 64, 32, 32, 1])
    step1 = step(train_params, x_train, y_train)
    assert type(step1) == tuple
    assert len(step1) == 4


def test_train():
    x_train = random.randint(random.PRNGKey(59), shape=(1000, 5), minval=0, 
                             maxval=50).astype(float)
    y_train = random.bernoulli(random.PRNGKey(997), 0.35, shape=(1000,)).astype(float)
    train_params = init_deep_fm(51, 5, [5, 64, 64, 32, 32, 1])
    out = train(train_params, x_train, y_train, 100)
    epochs, loss, accuracy, train_params = out
    assert len(out) == 4
    assert len(epochs) == 100
    assert loss[0] >= loss[99]
    assert accuracy[99] >= accuracy[0]


def test_predict():
    x_train = random.randint(random.PRNGKey(59), shape=(1000, 5), minval=0, 
                             maxval=50).astype(float)
    train_params = init_deep_fm(51, 5, [5, 64, 64, 32, 32, 1])
    predictions = predict(train_params, x_train[:5])
    assert type(predictions) == jaxlib.xla_extension.ArrayImpl
    assert predictions.shape == (5,)
