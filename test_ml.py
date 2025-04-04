import os
import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference


# Fixtures
# Create a small faux sample of data for testing
@pytest.fixture
def sample_data():
    """
    Fixture that provides a small, representative DataFrame mimicking
    the structure and types of the census dataset. Used for testing
    pipeline functions in isolation.
    """
    data = {
        "age": [25, 40],
        "workclass": ["Private", "Self-emp-not-inc"],
        "fnlwgt": [226802, 89814],
        "education": ["11th", "Bachelors"],
        "education-num": [7, 13],
        "marital-status": ["Never-married", "Married-civ-spouse"],
        "occupation": ["Machine-op-inspct", "Exec-managerial"],
        "relationship": ["Own-child", "Husband"],
        "race": ["Black", "White"],
        "sex": ["Male", "Male"],
        "capital-gain": [0, 0],
        "capital-loss": [0, 0],
        "hours-per-week": [40, 50],
        "native-country": ["United-States", "United-States"],
        "salary": ["<=50K", ">50K"]
    }
    return pd.DataFrame(data)


# Census.csv data
@pytest.fixture
def real_data():
    """
    Fixture that loads the real census dataset from the 'data' directory.
    Used to perform integration-level tests on real-world data, such as
    data validation and consistency checks.
    """
    project_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(project_path, "data", "census.csv")
    data = pd.read_csv(data_path, encoding='utf-8')
    return data


@pytest.fixture
def categorical_features():
    """
    Fixture that provides the list of categorical feature names used in
    preprocessing steps. These features are one-hot encoded during data
    transformation and must be consistently defined across tests.
    """
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]
    return cat_features


# Tests

# Functional Tests
# Test 1: process_data returns correct output types
def test_process_data_output_types(sample_data, categorical_features):
    """
    Tests that the `process_data` function returns the correct output types
    (numpy arrays for features and labels) and correct dimensions.
    Ensures preprocessing pipeline is functioning properly.
    """
    X, y, encoder, lb = process_data(
        sample_data, categorical_features=categorical_features, label="salary", training=True
    )
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert len(X) == len(y)
    assert X.ndim == 2
    assert y.ndim == 1


# Test 2: train_model returns a fitted model
def test_train_model_returns_model(sample_data, categorical_features):
    """
    Tests that the `train_model` function returns a fitted RandomForestClassifier
    with a working `predict` method, and that it can produce predictions with the
    correct shape.
    """
    X, y, _, _ = process_data(sample_data, categorical_features=categorical_features, label="salary", training=True)
    model, best_params = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "predict")
    preds = inference(model, X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == y.shape


# Test 3: compute_model_metrics returns expected metrics range
def test_compute_model_metrics_valid_range(sample_data, categorical_features):
    """
    Tests that the metrics returned by `compute_model_metrics` fall within
    the expected range [0, 1] for precision, recall, and F1 score.
    Validates correctness of metric computation.
    """
    X, y, _, _ = process_data(sample_data, categorical_features=categorical_features, label="salary", training=True)
    model, _ = train_model(X, y)
    preds = inference(model, X)
    precision, recall, f1 = compute_model_metrics(y, preds)

    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= f1 <= 1.0


# Data Integrity Tests

# Test 4: Age Range Check
def test_age_range(real_data):
    """
    Ensure all age values in the dataset are within a realistic human range.
    Ignores any missing values.
    """
    age_column = real_data["age"].dropna()
    assert age_column.between(0, 100).all(), "Age values should be between 0 and 100."


# Test 5: Salary Value Check
def test_salary_values(real_data):
    """
    Ensure the salary column only contains expected labels: '<=50K' or '>50K'.
    """
    valid_labels = {"<=50K", ">50K"}
    assert set(real_data["salary"].unique()).issubset(valid_labels), "Unexpected salary labels found."
