import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERICAL_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

ALL_FEATURE_COLS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def create_preprocessor() -> ColumnTransformer:
    """Build a Scikit-Learn ColumnTransformer for numerical and categorical features."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """Extract human-readable feature names after one-hot encoding."""
    num_names = NUMERICAL_FEATURES
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    return num_names + cat_names


if __name__ == "__main__":
    from data_loader import load_or_create_data

    df = load_or_create_data()
    prep = create_preprocessor()
    X = prep.fit_transform(df[ALL_FEATURE_COLS])
    feature_names = get_feature_names(prep)
    print("Preprocessed feature matrix shape:", X.shape)
    print("Number of transformed features:", len(feature_names))
    print("First 10 transformed feature names:", feature_names[:10])
