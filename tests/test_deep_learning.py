import pytest

torch = pytest.importorskip("torch")

from torch import nn
from torch.optim import Adam

from superalloy_ml.data import generate_synthetic_superalloy_dataset
from superalloy_ml.deep_learning.cnn import SimpleCNNClassifier
from superalloy_ml.deep_learning.datasets import (
    make_loaders_from_tensors,
    make_synthetic_image_dataset,
    make_synthetic_sequence_dataset,
    prepare_tabular_tensors,
    split_tensor_dataset,
)
from superalloy_ml.deep_learning.mlp import TabularMLP
from superalloy_ml.deep_learning.rnn_lstm import LSTMClassifier, RNNClassifier
from superalloy_ml.deep_learning.trainer import evaluate_model, train_model
from superalloy_ml.deep_learning.transformer import TransformerClassifier


def test_prepare_tabular_tensors_regression():
    df = generate_synthetic_superalloy_dataset(n_samples=80, random_state=42)

    bundle = prepare_tabular_tensors(
        df=df,
        target="creep_log_life_h",
        task="regression",
        test_size=0.25,
        random_state=42,
    )

    assert bundle.X_train.ndim == 2
    assert bundle.y_train.ndim == 2
    assert bundle.input_dim == bundle.X_train.shape[1]
    assert bundle.output_dim == 1


def test_prepare_tabular_tensors_classification():
    df = generate_synthetic_superalloy_dataset(n_samples=90, random_state=42)

    bundle = prepare_tabular_tensors(
        df=df,
        target="creep_life_class",
        task="classification",
        test_size=0.25,
        random_state=42,
    )

    assert bundle.X_train.ndim == 2
    assert bundle.y_train.ndim == 1
    assert bundle.output_dim >= 3


def test_mlp_forward_regression():
    model = TabularMLP(input_dim=10, output_dim=1)
    x = torch.randn(4, 10)

    y = model(x)

    assert y.shape == (4, 1)


def test_mlp_forward_classification():
    model = TabularMLP(input_dim=10, output_dim=3)
    x = torch.randn(4, 10)

    y = model(x)

    assert y.shape == (4, 3)


def test_cnn_forward():
    model = SimpleCNNClassifier(in_channels=1, n_classes=3)
    x = torch.randn(4, 1, 16, 16)

    y = model(x)

    assert y.shape == (4, 3)


def test_rnn_forward():
    model = RNNClassifier(input_dim=6, hidden_dim=16, n_classes=3)
    x = torch.randn(4, 20, 6)

    y = model(x)

    assert y.shape == (4, 3)


def test_lstm_forward():
    model = LSTMClassifier(input_dim=6, hidden_dim=16, n_classes=3)
    x = torch.randn(4, 20, 6)

    y = model(x)

    assert y.shape == (4, 3)


def test_transformer_forward():
    model = TransformerClassifier(
        input_dim=6,
        n_classes=3,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
    )
    x = torch.randn(4, 20, 6)

    y = model(x)

    assert y.shape == (4, 3)


def test_synthetic_image_dataset():
    X, y = make_synthetic_image_dataset(
        n_samples=30,
        image_size=16,
        n_classes=3,
        random_state=42,
    )

    assert X.shape == (30, 1, 16, 16)
    assert y.shape == (30,)


def test_synthetic_sequence_dataset():
    X, y = make_synthetic_sequence_dataset(
        n_samples=30,
        sequence_length=20,
        n_features=6,
        n_classes=3,
        random_state=42,
    )

    assert X.shape == (30, 20, 6)
    assert y.shape == (30,)


def test_train_small_mlp_classification():
    df = generate_synthetic_superalloy_dataset(n_samples=90, random_state=42)

    bundle = prepare_tabular_tensors(
        df=df,
        target="creep_life_class",
        task="classification",
        test_size=0.25,
        random_state=42,
    )

    train_loader, test_loader = make_loaders_from_tensors(
        bundle.X_train,
        bundle.y_train,
        bundle.X_test,
        bundle.y_test,
        batch_size=32,
    )

    model = TabularMLP(
        input_dim=bundle.input_dim,
        output_dim=bundle.output_dim,
        hidden_dims=[32],
        dropout=0.0,
    )

    optimizer = Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=1,
        task="classification",
        verbose=False,
    )

    metrics = evaluate_model(
        model=model,
        data_loader=test_loader,
        task="classification",
    )

    assert len(history.train_loss) == 1
    assert "accuracy" in metrics


def test_train_small_cnn():
    X, y = make_synthetic_image_dataset(
        n_samples=60,
        image_size=16,
        n_classes=3,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = split_tensor_dataset(
        X,
        y,
        test_size=0.25,
        random_state=42,
    )

    train_loader, test_loader = make_loaders_from_tensors(
        X_train,
        y_train,
        X_test,
        y_test,
        batch_size=16,
    )

    model = SimpleCNNClassifier(in_channels=1, n_classes=3, base_channels=8)
    optimizer = Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=1,
        task="classification",
        verbose=False,
    )

    assert len(history.train_loss) == 1