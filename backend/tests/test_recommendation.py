import pickle
import jaxlib
from pathlib import Path
import jaxlib.xla_extension
from shoulder.ml.ml.recommendation import preprocess, pretrain, finetune, recommend


def test_preprocess():
    path = "tests/test_data.pkl"
    with open(path, 'rb') as f:
        raw_data = pickle.load(f)

    # Making data for prediction only
    prediction_data = [{k: v for k, v in d.items() if k != "attended_event"} 
                       for d in raw_data]

    preprocessed_data_train = preprocess(raw_data)
    preprocessed_data_prediction = preprocess(prediction_data, predict=True)
    assert isinstance(preprocessed_data_train, tuple)
    assert isinstance(preprocessed_data_prediction, jaxlib.xla_extension.ArrayImpl)


def test_pretrain():
    path = "tests/test_data.pkl"
    with open(path, 'rb') as f:
        raw_data = pickle.load(f)
    out = pretrain(raw_data, path="tests/test_pretrain.pkl")
    epochs, _, _ = out
    assert len(out) == 3
    assert len(epochs) == 10
    assert Path('shoulder/ml/ml/figures/training_curves.jpg').is_file()
    assert Path(path).is_file()


def test_finetune():
    path = "tests/test_data.pkl"
    with open(path, 'rb') as f:
        raw_data = pickle.load(f)
    out = finetune(raw_data)
    assert len(out) == 3


def test_recommend():
    path = "tests/test_data.pkl"
    with open(path, 'rb') as f:
        raw_data = pickle.load(f)

    prediction_data = [{k: v for k, v in d.items() if k != "attended_event"} 
                       for d in raw_data]

    predictions = recommend(prediction_data)
    assert isinstance(predictions, jaxlib.xla_extension.ArrayImpl)
    assert len(predictions) == len(prediction_data)
