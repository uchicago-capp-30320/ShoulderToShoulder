from ml.train import step, train, predict
from ml.dataset import Dataset
from jax import random
import jaxlib
from ml.model import init_deep_fm, init_mlp_params
from pathlib import Path


def test_step():
    x_train = random.randint(random.PRNGKey(97), shape=(1000, 5), minval=0, 
                             maxval=50).astype(float)
    y_train = random.bernoulli(random.PRNGKey(997), 0.35, shape=(1000,)).astype(float)
    train_params = init_deep_fm(51, 5, 5)
    step1 = step(train_params, x_train, y_train)
    assert type(step1) == tuple
    assert len(step1) == 4



def test_train():
    x_train = random.randint(random.PRNGKey(706), shape=(1000, 5), minval=0, 
                             maxval=50).astype(float)
    y_train = random.bernoulli(random.PRNGKey(9970), 0.35, shape=(1000,)).astype(float)
    train_data = Dataset(x_train, y_train, 128, 127)
    train_params = init_deep_fm(51, 5, 5)
    out = train(train_params, train_data, 10, 32)
    epochs, loss, accuracy, train_params = out
    assert len(out) == 4
    assert len(epochs) == 10
    assert loss[0] > loss[9]
    assert accuracy[9] > accuracy[0]
    assert Path('ml/ml/figures/training_curves.jpg').is_file()


def test_predict():
    x_train = random.randint(random.PRNGKey(59), shape=(1000, 5), minval=0, 
                             maxval=50).astype(float)
    train_params = init_deep_fm(51, 5, 5)
    predictions = predict(train_params, x_train[:5])
    assert type(predictions) == jaxlib.xla_extension.ArrayImpl
    assert predictions.shape == (5, 1)
