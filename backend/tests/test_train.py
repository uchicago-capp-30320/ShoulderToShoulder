import os
import jaxlib
from jax import random
import jax.numpy as jnp
from pathlib import Path
from shoulder.ml.ml.dataset import Dataset
from shoulder.ml.ml.model import init_deep_fm
from shoulder.ml.ml.train import step, train, predict, save_outputs


def test_step():
    x_train = random.randint(random.PRNGKey(97), shape=(1000, 5), minval=0, 
                             maxval=50).astype(float)
    y_train = random.bernoulli(random.PRNGKey(997), 0.35, shape=(1000,)).astype(float)
    train_params = init_deep_fm(51, 5, 5)
    step1 = step(train_params, x_train, y_train)
    assert type(step1) == tuple
    assert len(step1) == 4


def test_save_outputs():
    accuracy = [0.95] * 10
    loss = [0.25] * 10
    epochs = [i + 1 for i in range(10)]
    train_params = init_deep_fm(51, 5, 5)
    path = 'tests/test_params.pkl'
    save_outputs(epochs, loss, accuracy, train_params, path)
    assert Path('shoulder/ml/ml/figures/training_curves.jpg').is_file()
    assert Path(path).is_file()


def test_train():
    path = 'tests/test_params.pkl'
    os.remove('shoulder/ml/ml/figures/training_curves.jpg')
    os.remove(path)
    x_train = random.randint(random.PRNGKey(706), shape=(1000, 5), minval=0, 
                             maxval=50).astype(float)
    y_train = random.bernoulli(random.PRNGKey(9970), 0.35, shape=(1000,)).astype(float)
    train_data = Dataset(x_train, y_train, 256, 127)
    train_params = init_deep_fm(51, 5, 5)
    out = train(train_params, train_data, 10, path)
    epochs, loss, accuracy, train_params = out
    assert len(out) == 4
    assert len(epochs) == 10
    assert loss[0] > loss[9]
    assert accuracy[9] > accuracy[0]
    assert Path('shoulder/ml/ml/figures/training_curves.jpg').is_file()
    assert Path(path).is_file()


def test_predict():
    x = random.randint(random.PRNGKey(59), shape=(5, 150), minval=0, maxval=50).astype(float)
    predictions = predict(x)
    assert type(predictions) == jaxlib.xla_extension.ArrayImpl
    assert predictions.shape == (5, 1)


def test_predict_new_user():
    # Testing embedding unseen users
    X = jnp.concat([random.randint(random.key(99), (1, 149), minval=0, maxval=50), 
                    jnp.array([[1000000]])], axis=1)
    assert not jnp.isnan(predict(X)).any()

