import pickle
import jaxlib
from pathlib import Path
import jaxlib.xla_extension
from shoulder.ml.ml.recommendation import preprocess, pretrain, finetune, recommend


def test_preprocess():
    path = "shoulder/tests/test_data.pkl"
    with open(path, 'rb') as f:
        raw_data = pickle.load(f)

    # Making data for prediction only
    prediction_data = [{k: v for k, v in d.items() if k != "attended_event"} 
                       for d in raw_data]

    preprocessed_data_train = preprocess(raw_data)
    preprocessed_data_prediction = preprocess(prediction_data, predict=True)
    assert isinstance(preprocessed_data_train, tuple), "Ensure output is a tuple"
    assert isinstance(preprocessed_data_prediction, jaxlib.xla_extension.ArrayImpl), "Ensure output is a jax array"


def test_pretrain():
    path = "shoulder/tests/test_data.pkl"
    with open(path, 'rb') as f:
        raw_data = pickle.load(f)
    out = pretrain(raw_data, path="shoulder/tests/test_pretrain.pkl")
    epochs, _, _ = out
    assert len(out) == 3, "Ensure output is a tuple"
    assert len(epochs) == 10, "Ensure correct number of epochs"
    assert Path('shoulder/ml/ml/figures/training_curves.jpg').is_file(), "Ensure training curve is saved"
    assert Path(path).is_file(), "Ensure model is saved"


def test_finetune():
    path = "shoulder/tests/test_data.pkl"
    with open(path, 'rb') as f:
        raw_data = pickle.load(f)
    out = finetune(raw_data)
    assert len(out) == 3, "Ensure output is a tuple"


def test_recommend():
    path = "shoulder/tests/test_data.pkl"
    with open(path, 'rb') as f:
        raw_data = pickle.load(f)

    prediction_data = [{k: v for k, v in d.items() if k != "attended_event"} 
                       for d in raw_data]

    predictions = recommend(prediction_data)
    assert isinstance(predictions, jaxlib.xla_extension.ArrayImpl), "Ensure output is a jax array"
    assert len(predictions) == len(prediction_data), "Ensure correct number of predictions"
