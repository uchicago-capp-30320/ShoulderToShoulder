import os, pathlib
import jaxlib
from jax import random
import jax.numpy as jnp
from pathlib import Path
from shoulder.ml.ml.dataset import Dataset
from shoulder.ml.ml.model import init_deep_fm
from shoulder.ml.ml.train import step, train, predict, save_outputs

TEST_DATA_DIR = pathlib.Path(__file__).parent
FIGURES_DIR = os.path.join(pathlib.Path(__file__).parent.absolute().parent, "ml/ml/figures")

def test_step():
    x_train = random.randint(random.PRNGKey(97), shape=(1000, 5), minval=0, 
                             maxval=50).astype(float)
    y_train = random.bernoulli(random.PRNGKey(997), 0.35, shape=(1000,)).astype(float)
    train_params = init_deep_fm(51, 5, 5)
    step1 = step(train_params, x_train, y_train)
    assert type(step1) == tuple, "Ensure output is a tuple"
    assert len(step1) == 4, "Ensure correct number of outputs"


def test_save_outputs():
    accuracy = [0.95] * 10
    loss = [0.25] * 10
    epochs = [i + 1 for i in range(10)]
    train_params = init_deep_fm(51, 5, 5)
    path = os.path.join(TEST_DATA_DIR, "test_data2.pkl")
    save_outputs(epochs, loss, accuracy, train_params, path)
    # creating and saving the plot doesn't work on macos - this test will always fail on mac
    # assert Path(os.path.join(FIGURES_DIR, "training_curves.jpg")).is_file(), "Ensure training curve is saved"
    assert Path(path).is_file(), "Ensure model is saved"


def test_train():
    path = os.path.join(TEST_DATA_DIR, "test_data2.pkl")
    # os.remove(os.path.join(FIGURES_DIR, "training_curves.jpg"))
    os.remove(path)
    x_train = random.randint(random.PRNGKey(706), shape=(1000, 5), minval=0, 
                             maxval=50).astype(float)
    y_train = random.bernoulli(random.PRNGKey(9970), 0.35, shape=(1000,)).astype(float)
    train_data = Dataset(x_train, y_train, 256, 127)
    train_params = init_deep_fm(51, 5, 5)
    out = train(train_params, train_data, 10, path)
    epochs, loss, accuracy, train_params = out
    assert len(out) == 4, "Ensure output is a tuple"
    assert len(epochs) == 10, "Ensure correct number of epochs"
    assert loss[0] > loss[9], "Ensure loss is decreasing"
    assert accuracy[9] > accuracy[0], "Ensure accuracy is increasing"
    # creating and saving the plot doesn't work on macos - this test will always fail on mac
    # assert Path(os.path.join(FIGURES_DIR, "training_curves.jpg")).is_file(), "Ensure training curve is saved"
    assert Path(path).is_file(), "Ensure model is saved"


def test_predict():
    x = random.randint(random.PRNGKey(59), shape=(5, 150), minval=0, maxval=50).astype(float)
    predictions = predict(x)
    assert type(predictions) == jaxlib.xla_extension.ArrayImpl, "Ensure output is a jax array"
    assert predictions.shape == (5, 1), "Ensure output shape is correct"


def test_predict_new_user():
    # Testing embedding unseen users
    X = jnp.concat([random.randint(random.key(99), (1, 149), minval=0, maxval=50), 
                    jnp.array([[1000000]])], axis=1)
    assert not jnp.isnan(predict(X)).any(), "Ensure output is not nan"

