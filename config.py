from pathlib import Path

# Project root
ROOT_DIR = Path(__file__).parent

# Data paths
RAW_DATA_PATH = ROOT_DIR / "data" / "raw.csv"
CLEANED_DATA_PATH = ROOT_DIR / "data" / "cleaned_data.csv"
PREDICTIONS_OUTPUT_PATH = ROOT_DIR / "data" / "predictions_output.csv"

# Model paths
MODEL_PATH = ROOT_DIR / "main" / "svm_best_pipeline.joblib"
CONFUSION_MATRIX_PATH = ROOT_DIR / "main" / "confusion_matrix.png"
FEATURE_IMPORTANCE_PATH = ROOT_DIR / "main" / "feature_importance.png"

# Feature definitions
CATEGORICAL_FEATURES = ["workclass", "education", "marital_status", "occupation", "relationship", "race", "sex", "native_country"]
NUMERICAL_FEATURES = ["age", "education_num", "fnlwgt", "hours_per_week"]
TARGET_COLUMN = "income"
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

# Preprocessing
IQR_MULTIPLIER = 1.5
LOW_VARIANCE_COLUMNS = ["capital_gain", "capital_loss"]
AGE_BINS = [10, 20, 30, 40, 50, 60, 70, 80]
AGE_LABELS = ["10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79"]

# Model training
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5
N_ITER = 10

# SVM hyperparameter grid
SVM_PARAM_GRID = {
    "classifier__C": [0.01, 0.1, 1, 10, 100],
    "classifier__kernel": ["linear", "rbf", "poly"],
    "classifier__gamma": ["scale", "auto"]
}
