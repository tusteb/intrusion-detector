import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import model_utils

def test_predict_from_df():
    fake_model = Mock()
    fake_model.predict.return_value = np.array([0, 1])
    fake_model.predict_proba.return_value = np.array([[0.2, 0.8], [0.9, 0.1]])

    fake_encoder = Mock()
    fake_encoder.inverse_transform.side_effect = lambda x: ['Normal' if i == 0 else 'Malicious' for i in x]

    selected_features = ['f1', 'f2']
    df = pd.DataFrame({'f1': [1, 2], 'f2': [3, 4]})

    labels, probs, confidences = model_utils.predict_from_df(df, fake_model, fake_encoder, selected_features)

    assert list(labels) == ['Normal', 'Malicious']
    assert confidences[0] == 0.8
    assert confidences[1] == 0.9


def test_predict_single():
    fake_model = Mock()
    fake_model.predict_proba.return_value = np.array([[0.1, 0.9]])

    fake_encoder = Mock()
    fake_encoder.inverse_transform.return_value = ['Malicious']

    selected_features = ['f1', 'f2']
    features = {'f1': 10, 'f2': 20}

    label, probs, confidence = model_utils.predict_single(features, fake_model, fake_encoder, selected_features)

    assert label == 'Malicious'
    assert confidence == 0.9
    assert probs[1] == 0.9


def test_missing_feature_raises_keyerror():
    selected_features = ['f1', 'f2']
    df = pd.DataFrame([{'f1': 10}])  # отсутствует f2

    with pytest.raises(KeyError):

        _ = df[selected_features]
