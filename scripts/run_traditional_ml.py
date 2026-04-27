from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from superalloy_ml.constants import TARGET_COLUMNS
from superalloy_ml.data import load_configured_dataset, read_table, split_features_targets
from superalloy_ml.features import make_features
from superalloy_ml.reporting import (
    create_output_dirs,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_prediction_vs_actual,
    plot_residuals,
    save_table,
    write_markdown_report,
)
from superalloy_ml.traditional_ml import (
    cross_validate_models,
    get_best_model_name,
    train_test_evaluate_models,
)
from superalloy_ml.utils import get_config, print_section


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行传统机器学习模型。")
    parser.add_argument("--config", default="configs/default.yaml", help="配置文件路径。")
    parser.add_argument("--data-path", default=None, help="外部数据路径，支持 CSV 或 Excel。")
    parser.add_argument("--regression-target", default=None, help="回归目标列。")
    parser.add_argument("--classification-target", default=None, help="分类目标列。")
    parser.add_argument("--skip-regression", action="store_true", help="跳过回归任务。")
    parser.add_argument("--skip-classification", action="store_true", help="跳过分类任务。")
    parser.add_argument("--no-xgboost", action="store_true", help="不使用 XGBoost。")
    return parser.parse_args()


def load_input_dataframe(args: argparse.Namespace, config: dict) -> pd.DataFrame:
    if args.data_path:
        return read_table(args.data_path)

    return load_configured_dataset(config)


def apply_feature_engineering(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    feature_config = config.get("features", {})

    return make_features(
        df,
        normalize=bool(feature_config.get("normalize_composition", True)),
        add_composition=bool(feature_config.get("add_composition_features", True)),
        add_physical=bool(feature_config.get("add_physical_descriptors", True)),
        add_processing=bool(feature_config.get("add_processing_features", True)),
        add_interactions=bool(feature_config.get("add_interaction_features", True)),
    )


def run_regression(
    df: pd.DataFrame,
    target: str,
    config: dict,
    output_dirs: dict[str, Path],
    include_xgboost: bool,
) -> None:
    print_section("传统机器学习回归任务")

    if target not in df.columns:
        print(f"未找到回归目标列，跳过回归任务: {target}")
        return

    X, y = split_features_targets(df, target=target)

    modeling_config = config.get("modeling", {})
    preprocessing_config = config.get("preprocessing", {})
    data_config = config.get("data", {})

    save_models = bool(modeling_config.get("save_models", True))
    generate_figures = bool(modeling_config.get("generate_figures", True))

    model_dir = output_dirs["models"] / "traditional_regression" if save_models else None

    results, trained_models = train_test_evaluate_models(
        X=X,
        y=y,
        task="regression",
        test_size=float(data_config.get("test_size", 0.2)),
        random_state=int(config.get("project", {}).get("random_state", 42)),
        include_xgboost=include_xgboost,
        scaler=preprocessing_config.get("scaler", "standard"),
        numeric_strategy=preprocessing_config.get("numeric_impute_strategy", "median"),
        categorical_strategy=preprocessing_config.get("categorical_impute_strategy", "most_frequent"),
        save_model_dir=model_dir,
    )

    cv_results = cross_validate_models(
        X=X,
        y=y,
        task="regression",
        cv=int(modeling_config.get("cv_folds", 5)),
        random_state=int(config.get("project", {}).get("random_state", 42)),
        include_xgboost=include_xgboost,
        scaler=preprocessing_config.get("scaler", "standard"),
        numeric_strategy=preprocessing_config.get("numeric_impute_strategy", "median"),
        categorical_strategy=preprocessing_config.get("categorical_impute_strategy", "most_frequent"),
    )

    save_table(results, output_dirs["reports"] / "traditional_regression_test_results.csv")
    save_table(cv_results, output_dirs["reports"] / "traditional_regression_cv_results.csv")

    print("测试集结果:")
    print(results)

    print()
    print("交叉验证结果:")
    print(cv_results)

    best_model_name = get_best_model_name(results, task="regression")
    best_model = trained_models[best_model_name]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(data_config.get("test_size", 0.2)),
        random_state=int(config.get("project", {}).get("random_state", 42)),
    )
    _ = X_train, y_train

    y_pred = best_model.predict(X_test)

    prediction_df = pd.DataFrame(
        {
            "y_true": y_test.to_numpy(),
            "y_pred": y_pred,
            "residual": y_test.to_numpy() - y_pred,
        }
    )
    save_table(prediction_df, output_dirs["predictions"] / "traditional_regression_predictions.csv")

    if generate_figures:
        plot_model_comparison(
            results=results,
            metric="r2",
            path=output_dirs["figures"] / "traditional_regression_r2.png",
            title="回归模型 R2 对比",
            higher_is_better=True,
        )

        plot_model_comparison(
            results=results,
            metric="rmse",
            path=output_dirs["figures"] / "traditional_regression_rmse.png",
            title="回归模型 RMSE 对比",
            higher_is_better=False,
        )

        plot_prediction_vs_actual(
            y_true=y_test.to_numpy(),
            y_pred=y_pred,
            path=output_dirs["figures"] / "traditional_regression_prediction_vs_actual.png",
            title=f"最佳回归模型预测效果：{best_model_name}",
        )

        plot_residuals(
            y_true=y_test.to_numpy(),
            y_pred=y_pred,
            path=output_dirs["figures"] / "traditional_regression_residuals.png",
            title=f"最佳回归模型残差图：{best_model_name}",
        )

    print()
    print(f"最佳回归模型: {best_model_name}")


def run_classification(
    df: pd.DataFrame,
    target: str,
    config: dict,
    output_dirs: dict[str, Path],
    include_xgboost: bool,
) -> None:
    print_section("传统机器学习分类任务")

    if target not in df.columns:
        print(f"未找到分类目标列，跳过分类任务: {target}")
        return

    X, y = split_features_targets(df, target=target)

    modeling_config = config.get("modeling", {})
    preprocessing_config = config.get("preprocessing", {})
    data_config = config.get("data", {})

    save_models = bool(modeling_config.get("save_models", True))
    generate_figures = bool(modeling_config.get("generate_figures", True))

    model_dir = output_dirs["models"] / "traditional_classification" if save_models else None

    results, trained_models = train_test_evaluate_models(
        X=X,
        y=y,
        task="classification",
        test_size=float(data_config.get("test_size", 0.2)),
        random_state=int(config.get("project", {}).get("random_state", 42)),
        include_xgboost=include_xgboost,
        scaler=preprocessing_config.get("scaler", "standard"),
        numeric_strategy=preprocessing_config.get("numeric_impute_strategy", "median"),
        categorical_strategy=preprocessing_config.get("categorical_impute_strategy", "most_frequent"),
        save_model_dir=model_dir,
    )

    cv_results = cross_validate_models(
        X=X,
        y=y,
        task="classification",
        cv=int(modeling_config.get("cv_folds", 5)),
        random_state=int(config.get("project", {}).get("random_state", 42)),
        include_xgboost=include_xgboost,
        scaler=preprocessing_config.get("scaler", "standard"),
        numeric_strategy=preprocessing_config.get("numeric_impute_strategy", "median"),
        categorical_strategy=preprocessing_config.get("categorical_impute_strategy", "most_frequent"),
    )

    save_table(results, output_dirs["reports"] / "traditional_classification_test_results.csv")
    save_table(cv_results, output_dirs["reports"] / "traditional_classification_cv_results.csv")

    print("测试集结果:")
    print(results)

    print()
    print("交叉验证结果:")
    print(cv_results)

    best_model_name = get_best_model_name(results, task="classification")
    best_model = trained_models[best_model_name]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(data_config.get("test_size", 0.2)),
        random_state=int(config.get("project", {}).get("random_state", 42)),
        stratify=y,
    )
    _ = X_train, y_train

    y_pred = best_model.predict(X_test)

    prediction_df = pd.DataFrame(
        {
            "y_true": y_test.to_numpy(),
            "y_pred": y_pred,
        }
    )
    save_table(
        prediction_df,
        output_dirs["predictions"] / "traditional_classification_predictions.csv",
    )

    if generate_figures:
        plot_model_comparison(
            results=results,
            metric="f1",
            path=output_dirs["figures"] / "traditional_classification_f1.png",
            title="分类模型 F1 对比",
            higher_is_better=True,
        )

        plot_model_comparison(
            results=results,
            metric="accuracy",
            path=output_dirs["figures"] / "traditional_classification_accuracy.png",
            title="分类模型 Accuracy 对比",
            higher_is_better=True,
        )

        plot_confusion_matrix(
            y_true=y_test.to_numpy(),
            y_pred=y_pred,
            path=output_dirs["figures"] / "traditional_classification_confusion_matrix.png",
            title=f"最佳分类模型混淆矩阵：{best_model_name}",
        )

    print()
    print(f"最佳分类模型: {best_model_name}")


def main() -> None:
    args = parse_args()
    config = get_config(args.config)

    print_section("运行传统机器学习流程")

    output_base_dir = config.get("outputs", {}).get("base_dir", "outputs")
    output_dirs = create_output_dirs(ROOT / output_base_dir)

    df = load_input_dataframe(args, config)
    df = apply_feature_engineering(df, config)

    include_xgboost = bool(config.get("modeling", {}).get("use_xgboost", True))
    include_xgboost = include_xgboost and not args.no_xgboost

    regression_target = args.regression_target or config.get("target", {}).get(
        "regression",
        "creep_log_life_h",
    )
    classification_target = args.classification_target or config.get("target", {}).get(
        "classification",
        "creep_life_class",
    )

    if not args.skip_regression:
        run_regression(
            df=df,
            target=regression_target,
            config=config,
            output_dirs=output_dirs,
            include_xgboost=include_xgboost,
        )

    if not args.skip_classification:
        run_classification(
            df=df,
            target=classification_target,
            config=config,
            output_dirs=output_dirs,
            include_xgboost=include_xgboost,
        )

    write_markdown_report(
        path=output_dirs["reports"] / "traditional_ml_report.md",
        title="传统机器学习结果报告",
        sections={
            "数据说明": "数据经过读取、特征工程和预处理后进入模型训练流程。",
            "模型范围": (
                "包含线性模型、KNN、SVM/SVR、决策树、随机森林、Extra Trees、"
                "梯度提升树和 XGBoost。"
            ),
            "输出说明": (
                "模型结果表、交叉验证结果、预测结果、模型文件和图像文件均保存在 outputs 目录。"
            ),
        },
    )

    print_section("传统机器学习流程完成")
    print(f"输出目录: {output_dirs['base']}")


if __name__ == "__main__":
    main()