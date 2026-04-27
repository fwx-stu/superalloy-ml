from superalloy_ml.data import generate_synthetic_superalloy_dataset, split_features_targets
from superalloy_ml.preprocessing import (
    create_preprocessor,
    get_categorical_columns,
    get_feature_names,
    get_numeric_columns,
    preprocess_fit_transform,
    preprocess_transform,
)


def test_get_numeric_columns():
    df = generate_synthetic_superalloy_dataset(n_samples=20, random_state=42)
    numeric_columns = get_numeric_columns(df)

    assert len(numeric_columns) > 0


def test_get_categorical_columns():
    df = generate_synthetic_superalloy_dataset(n_samples=20, random_state=42)
    categorical_columns = get_categorical_columns(df)

    assert isinstance(categorical_columns, list)


def test_create_preprocessor():
    df = generate_synthetic_superalloy_dataset(n_samples=30, random_state=42)
    X, _ = split_features_targets(df, target="creep_log_life_h")

    preprocessor = create_preprocessor(X)

    assert preprocessor is not None


def test_preprocess_fit_transform():
    df = generate_synthetic_superalloy_dataset(n_samples=30, random_state=42)
    X, _ = split_features_targets(df, target="creep_log_life_h")

    X_processed, preprocessor = preprocess_fit_transform(X)

    assert X_processed.shape[0] == X.shape[0]
    assert preprocessor is not None


def test_preprocess_transform():
    df = generate_synthetic_superalloy_dataset(n_samples=30, random_state=42)
    X, _ = split_features_targets(df, target="creep_log_life_h")

    X_processed, preprocessor = preprocess_fit_transform(X)
    X_processed_again = preprocess_transform(X, preprocessor)

    assert X_processed_again.shape == X_processed.shape


def test_get_feature_names():
    df = generate_synthetic_superalloy_dataset(n_samples=30, random_state=42)
    X, _ = split_features_targets(df, target="creep_log_life_h")

    _, preprocessor = preprocess_fit_transform(X)
    names = get_feature_names(preprocessor)

    assert isinstance(names, list)