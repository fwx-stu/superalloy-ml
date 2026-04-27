from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 PyTorch 深度学习模型。")
    parser.add_argument("--config", default="configs/default.yaml", help="配置文件路径。")
    parser.add_argument("--data-path", default=None, help="外部数据路径，支持 CSV 或 Excel。")
    parser.add_argument("--epochs", type=int, default=8, help="训练轮数。")
    parser.add_argument("--batch-size", type=int, default=64, help="批大小。")
    parser.add_argument("--skip-tabular", action="store_true", help="跳过表格 MLP。")
    parser.add_argument("--skip-image", action="store_true", help="跳过 CNN 图像示例。")
    parser.add_argument("--skip-sequence", action="store_true", help="跳过序列模型。")
    return parser.parse_args()


def main() -> None:
    try:
        import torch
        from torch import nn
        from torch.optim import Adam
    except Exception as exc:
        print("当前环境未安装 PyTorch。请先安装 requirements-dl.txt。")
        print(f"错误信息: {exc}")
        return

    from superalloy_ml.data import load_configured_dataset, read_table
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
    from superalloy_ml.deep_learning.trainer import (
        evaluate_model,
        get_device,
        save_torch_model,
        set_torch_seed,
        train_model,
    )
    from superalloy_ml.deep_learning.transformer import TransformerClassifier
    from superalloy_ml.features import make_features
    from superalloy_ml.reporting import create_output_dirs, save_table
    from superalloy_ml.utils import get_config, print_section

    args = parse_args()
    config = get_config(args.config)

    print_section("运行深度学习模型")

    project_config = config.get("project", {})
    data_config = config.get("data", {})
    preprocessing_config = config.get("preprocessing", {})
    feature_config = config.get("features", {})
    output_base_dir = config.get("outputs", {}).get("base_dir", "outputs")

    random_state = int(project_config.get("random_state", 42))
    test_size = float(data_config.get("test_size", 0.2))

    set_torch_seed(random_state)
    device = get_device(prefer_gpu=True)

    print(f"使用设备: {device}")

    output_dirs = create_output_dirs(ROOT / output_base_dir)
    rows = []

    if not args.skip_tabular:
        print_section("表格 MLP 回归任务")

        if args.data_path:
            df = read_table(args.data_path)
        else:
            df = load_configured_dataset(config)

        df = make_features(
            df,
            normalize=bool(feature_config.get("normalize_composition", True)),
            add_composition=bool(feature_config.get("add_composition_features", True)),
            add_physical=bool(feature_config.get("add_physical_descriptors", True)),
            add_processing=bool(feature_config.get("add_processing_features", True)),
            add_interactions=bool(feature_config.get("add_interaction_features", True)),
        )

        regression_target = config.get("target", {}).get("regression", "creep_log_life_h")
        classification_target = config.get("target", {}).get("classification", "creep_life_class")

        if regression_target in df.columns:
            bundle = prepare_tabular_tensors(
                df=df,
                target=regression_target,
                task="regression",
                test_size=test_size,
                random_state=random_state,
                scaler=preprocessing_config.get("scaler", "standard"),
            )

            train_loader, test_loader = make_loaders_from_tensors(
                bundle.X_train,
                bundle.y_train,
                bundle.X_test,
                bundle.y_test,
                batch_size=args.batch_size,
            )

            model = TabularMLP(
                input_dim=bundle.input_dim,
                output_dim=1,
                hidden_dims=[128, 64],
                dropout=0.1,
            )

            optimizer = Adam(model.parameters(), lr=1e-3)
            loss_fn = nn.MSELoss()

            history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=test_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epochs=args.epochs,
                device=device,
                task="regression",
                verbose=True,
            )

            metrics = evaluate_model(
                model=model,
                data_loader=test_loader,
                device=device,
                task="regression",
            )

            row = {
                "model_name": "mlp_regressor",
                "task": "regression",
                "final_train_loss": history.train_loss[-1],
                "final_val_loss": history.val_loss[-1],
            }
            row.update(metrics)
            rows.append(row)

            save_torch_model(
                model,
                output_dirs["models"] / "mlp_regressor.pt",
            )

        print_section("表格 MLP 分类任务")

        if classification_target in df.columns:
            bundle = prepare_tabular_tensors(
                df=df,
                target=classification_target,
                task="classification",
                test_size=test_size,
                random_state=random_state,
                scaler=preprocessing_config.get("scaler", "standard"),
            )

            train_loader, test_loader = make_loaders_from_tensors(
                bundle.X_train,
                bundle.y_train,
                bundle.X_test,
                bundle.y_test,
                batch_size=args.batch_size,
            )

            model = TabularMLP(
                input_dim=bundle.input_dim,
                output_dim=bundle.output_dim,
                hidden_dims=[128, 64],
                dropout=0.1,
            )

            optimizer = Adam(model.parameters(), lr=1e-3)
            loss_fn = nn.CrossEntropyLoss()

            history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=test_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epochs=args.epochs,
                device=device,
                task="classification",
                verbose=True,
            )

            metrics = evaluate_model(
                model=model,
                data_loader=test_loader,
                device=device,
                task="classification",
            )

            row = {
                "model_name": "mlp_classifier",
                "task": "classification",
                "final_train_loss": history.train_loss[-1],
                "final_val_loss": history.val_loss[-1],
            }
            row.update(metrics)
            rows.append(row)

            save_torch_model(
                model,
                output_dirs["models"] / "mlp_classifier.pt",
            )

    if not args.skip_image:
        print_section("CNN 图像分类示例")

        X_img, y_img = make_synthetic_image_dataset(
            n_samples=300,
            image_size=16,
            n_classes=3,
            random_state=random_state,
        )

        X_train, X_test, y_train, y_test = split_tensor_dataset(
            X_img,
            y_img,
            test_size=0.2,
            random_state=random_state,
        )

        train_loader, test_loader = make_loaders_from_tensors(
            X_train,
            y_train,
            X_test,
            y_test,
            batch_size=args.batch_size,
        )

        model = SimpleCNNClassifier(in_channels=1, n_classes=3)
        optimizer = Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()

        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epochs=args.epochs,
            device=device,
            task="classification",
            verbose=True,
        )

        metrics = evaluate_model(
            model=model,
            data_loader=test_loader,
            device=device,
            task="classification",
        )

        row = {
            "model_name": "cnn_classifier",
            "task": "classification",
            "final_train_loss": history.train_loss[-1],
            "final_val_loss": history.val_loss[-1],
        }
        row.update(metrics)
        rows.append(row)

        save_torch_model(
            model,
            output_dirs["models"] / "cnn_classifier.pt",
        )

    if not args.skip_sequence:
        print_section("序列模型分类示例")

        X_seq, y_seq = make_synthetic_sequence_dataset(
            n_samples=300,
            sequence_length=20,
            n_features=6,
            n_classes=3,
            random_state=random_state,
        )

        X_train, X_test, y_train, y_test = split_tensor_dataset(
            X_seq,
            y_seq,
            test_size=0.2,
            random_state=random_state,
        )

        train_loader, test_loader = make_loaders_from_tensors(
            X_train,
            y_train,
            X_test,
            y_test,
            batch_size=args.batch_size,
        )

        sequence_models = {
            "rnn_classifier": RNNClassifier(
                input_dim=6,
                hidden_dim=48,
                n_classes=3,
            ),
            "lstm_classifier": LSTMClassifier(
                input_dim=6,
                hidden_dim=48,
                n_classes=3,
            ),
            "transformer_classifier": TransformerClassifier(
                input_dim=6,
                n_classes=3,
                d_model=48,
                nhead=4,
                num_layers=2,
                dim_feedforward=96,
            ),
        }

        for model_name, model in sequence_models.items():
            print_section(model_name)

            optimizer = Adam(model.parameters(), lr=1e-3)
            loss_fn = nn.CrossEntropyLoss()

            history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=test_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epochs=args.epochs,
                device=device,
                task="classification",
                verbose=True,
            )

            metrics = evaluate_model(
                model=model,
                data_loader=test_loader,
                device=device,
                task="classification",
            )

            row = {
                "model_name": model_name,
                "task": "classification",
                "final_train_loss": history.train_loss[-1],
                "final_val_loss": history.val_loss[-1],
            }
            row.update(metrics)
            rows.append(row)

            save_torch_model(
                model,
                output_dirs["models"] / f"{model_name}.pt",
            )

    results = pd.DataFrame(rows)
    save_table(results, output_dirs["reports"] / "deep_learning_results.csv")

    print_section("深度学习流程完成")
    print(results)
    print(f"输出目录: {output_dirs['base']}")


if __name__ == "__main__":
    main()