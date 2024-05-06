<<<<<<< HEAD
import os
import jaxlib
from jax import random
from pathlib import Path
from ml.dataset import Dataset
from ml.model import init_deep_fm
from ml.train import step, train, predict, save_outputs
=======
from ml.train import step, train, predict
from ml.dataset import Dataset
from jax import random
import jaxlib
from ml.model import init_deep_fm
from pathlib import Path
>>>>>>> 03e10757a2d533e24b95eb4b5551f20a6dc8cf1e


def test_step():
    x_train = random.randint(random.PRNGKey(97), shape=(1000, 5), minval=0, 
                             maxval=50).astype(float)
    y_train = random.bernoulli(random.PRNGKey(997), 0.35, shape=(1000,)).astype(float)
<<<<<<< HEAD
    train_params = init_deep_fm(51, 5, 5)
=======
    train_params = init_deep_fm(51, 5, [5, 64, 64, 32, 32, 1])
>>>>>>> 03e10757a2d533e24b95eb4b5551f20a6dc8cf1e
    step1 = step(train_params, x_train, y_train)
    assert type(step1) == tuple
    assert len(step1) == 4


<<<<<<< HEAD
def test_save_outputs():
    accuracy = [0.95] * 10
    loss = [0.25] * 10
    epochs = [i + 1 for i in range(10)]
    train_params = init_deep_fm(51, 5, 5)
    save_outputs(epochs, loss, accuracy, train_params)
    assert Path('ml/ml/figures/training_curves.jpg').is_file()
    assert Path('ml/ml/weights/parameters.pkl').is_file()


def test_train():
    os.remove('ml/ml/figures/training_curves.jpg')
    x_train = random.randint(random.PRNGKey(706), shape=(1000, 5), minval=0, 
                             maxval=50).astype(float)
    y_train = random.bernoulli(random.PRNGKey(9970), 0.35, shape=(1000,)).astype(float)
    train_data = Dataset(x_train, y_train, 256, 127)
    train_params = init_deep_fm(51, 5, 5)
    out = train(train_params, train_data, 10)
    epochs, loss, accuracy, train_params = out
    assert len(out) == 4
    assert len(epochs) == 10
    assert loss[0] > loss[9]
    assert accuracy[9] > accuracy[0]
    assert Path('ml/ml/figures/training_curves.jpg').is_file()
    assert Path('ml/ml/weights/parameters.pkl').is_file()


def test_predict():
    x = random.randint(random.PRNGKey(59), shape=(5, 5), minval=0, maxval=50).astype(float)
    predictions = predict(x)
    assert type(predictions) == jaxlib.xla_extension.ArrayImpl
    assert predictions.shape == (5, 1)

=======
def test_train():
    x_train = random.randint(random.PRNGKey(706), shape=(1000, 5), minval=0, 
                             maxval=50).astype(float)
    y_train = random.bernoulli(random.PRNGKey(9970), 0.35, shape=(1000,)).astype(float)
    train_data = Dataset(x_train, y_train, 128, 90)
    train_params = init_deep_fm(51, 5, [5, 64, 64, 32, 32, 1])
    out = train(train_params, train_data, 100)
    epochs, loss, accuracy, train_params = out
    assert len(out) == 4
    assert len(epochs) == 100
    assert loss[0] >= loss[99]
    assert accuracy[99] >= accuracy[0]
    assert Path('ml/figures/training_curves.jpg').is_file()


def test_predict():
    x_train = random.randint(random.PRNGKey(59), shape=(1000, 5), minval=0, 
                             maxval=50).astype(float)
    train_params = init_deep_fm(51, 5, [5, 64, 64, 32, 32, 1])
    predictions = predict(train_params, x_train[:5])
    assert type(predictions) == jaxlib.xla_extension.ArrayImpl
    assert predictions.shape == (5,)
>>>>>>> 03e10757a2d533e24b95eb4b5551f20a6dc8cf1e
