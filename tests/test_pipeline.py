import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_df():
    """Minimal synthetic DataFrame matching expected schema."""
    return pd.DataFrame({
        "age": [25, 40, 35, 52, 28],
        "workclass": ["Private", "Self-emp", "Private", "Gov", "Private"],
        "fnlwgt": [200000, 150000, 180000, 220000, 170000],
        "education": ["Bachelors", "HS-grad", "Masters", "Bachelors", "HS-grad"],
        "education_num": [13, 9, 14, 13, 9],
        "marital_status": ["Never-married", "Married", "Divorced", "Married", "Never-married"],
        "occupation": ["Tech-support", "Craft-repair", "Exec-managerial", "Prof-specialty", "Sales"],
        "relationship": ["Own-child", "Husband", "Not-in-family", "Husband", "Own-child"],
        "race": ["White", "Black", "White", "Asian", "White"],
        "sex": ["Male", "Male", "Female", "Male", "Female"],
        "capital_gain": [0, 0, 5000, 0, 0],
        "capital_loss": [0, 0, 0, 0, 0],
        "hours_per_week": [40, 45, 50, 40, 35],
        "native_country": ["United-States"] * 5,
        "income": ["<=50K", ">50K", ">50K", "<=50K", "<=50K"],
    })


def test_hypothesis_results_schema(sample_df):
    from main.hypothesis_testing import test_gender_income, test_education_hours
    result = test_gender_income(sample_df)
    assert "p_value" in result
    assert "effect_size" in result
    assert 0 <= result["p_value"] <= 1
    assert result["result"] in ["Reject H0", "Fail to reject H0"]


def test_education_hours_raises_on_missing_value(sample_df):
    from main.hypothesis_testing import test_education_hours
    bad_df = sample_df.copy()
    bad_df["education"] = "Other"
    with pytest.raises(ValueError, match="not found in data"):
        test_education_hours(bad_df)


def test_load_features_raises_on_missing_columns(tmp_path):
    from main.ml_modeling import load_features
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"col1": [1, 2]}).to_csv(bad_csv, index=False)
    with pytest.raises(ValueError, match="Missing columns"):
        load_features(bad_csv)


def test_predict_new_raises_on_missing_fields():
    """Verify predict_new raises ValueError for incomplete input dict."""
    from main.ml_modeling import predict_new
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES

    # Build a minimal mock pipeline that won't require the real model file
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERICAL_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])
    mock_model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", SVC(probability=True)),
    ])
    # Mock fit on a tiny DataFrame so it can predict
    import pandas as pd
    import numpy as np
    X_fake = pd.DataFrame({
        "workclass": ["Private", "Private"], "education": ["Bachelors", "HS-grad"],
        "marital_status": ["Never-married", "Married"], "occupation": ["Tech-support", "Sales"],
        "relationship": ["Own-child", "Husband"], "race": ["White", "White"],
        "sex": ["Male", "Female"], "native_country": ["United-States", "United-States"],
        "age": [30, 45], "education_num": [13, 9], "fnlwgt": [100000, 120000],
        "hours_per_week": [40, 45],
    })
    y_fake = [">50K", "<=50K"]
    mock_model.fit(X_fake, y_fake)

    with pytest.raises(ValueError, match="missing required fields"):
        predict_new(mock_model, {"age": 30})  # incomplete input — missing categorical fields
