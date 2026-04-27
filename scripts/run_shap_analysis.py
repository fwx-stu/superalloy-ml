from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行模型解释分析。")
    parser.add_argument("--config", default="configs/default.yaml", help="配置文件路径。")
    parser.add_argument("--data-path", default=None, help="外部数据路径，支持 CSV 或 Excel。")
    parser.add_argument(
        "--task",
        choices=["regression", "classification"],
        default="regression",
        help="任务类型。",
    )
    parser.add_argument("--target", default=None, help="目标列名称。")
    parser.add_argument("--model-name", default=None, help="指定模型名称。")
    parser.add_argument("--max-samples", type=int, default=200, help="SHAP 最大样本数量。")
    parser.add_argument("--skip-shap", action="store_true", help="跳过 SHAP，只运行其他解释方法。")
    return parser.parse_args()


def main() -> None:
    from superalloy_ml.data import load_configured_dataset, read_table, split_features_targets
    from superalloy_ml.explainability import (
        compute_permutation_importance,
        compute_shap_summary,
        get_builtin_feature_importance,
        plot_importance_bar,
        save_explainability_tables,
    )
    from superalloy_ml.features import make_features
    from superalloy_ml.reporting import create_output_dirs
    from superalloy_ml.traditional_ml import build_pipeline, get_models
    from superalloy_ml.utils import get_config, print_section

    args = parse_args()
    config = get_config(args.config)

    print_section("运行模型解释分析")

    if args.data_path:
        df = read_table(args.data_path)
    else:
        df = load_configured_dataset(config)

    feature_config = config.get("features", {})
    preprocessing_config = config.get("preprocessing", {})
    modeling_config = config.get("modeling", {})
    project_config = config.get("project", {})
    data_config = config.get("data", {})
    output_config = config.get("outputs", {})

    df = make_features(
        df,
        normalize=bool(feature_config.get("normalize_composition", True)),
        add_composition=bool(feature_config.get("add_composition_features", True)),
        add_physical=bool(feature_config.get("add_physical_descriptors", True)),
        add_processing=bool(feature_config.get("add_processing_features", True)),
        add_interactions=bool(feature_config.get("add_interaction_features", True)),
    )

    if args.target is not None:
        target = args.target
    elif args.task == "regression":
        target = config.get("target", {}).get("regression", "creep_log_life_h")
    else:
        target = config.get("target", {}).get("classification", "creep_life_class")

    if target not in df.columns:
        raise ValueError(f"目标列不存在: {target}")

    X, y = split_features_targets(df, target=target)

    include_xgboost = bool(modeling_config.get("use_xgboost", True))
    random_state = int(project_config.get("random_state", 42))
    test_size = float(data_config.get("test_size", 0.2))

    models = get_models(
        task=args.task,
        random_state=random_state,
        include_xgboost=include_xgboost,
    )

    if args.model_name is None:
        if args.task == "regression":
            preferred = [
                "xgboost_regressor",
                "random_forest_regressor",
                "gradient_boosting_regressor",
            ]
        else:
            preferred = [
                "xgboost_classifier",
                "random_forest_classifier",
                "gradient_boosting_classifier",
            ]

        model_name = next((name for name in preferred if name in models), list(models.keys())[0])
    else:
        model_name = args.model_name

    if model_name not in models:
        raise ValueError(f"模型不存在: {model_name}。可选模型: {list(models)}")

    print(f"任务类型: {args.task}")
    print(f"目标列: {target}")
    print(f"解释模型: {model_name}")

    stratify = y if args.task == "classification" else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    pipeline = build_pipeline(
        X_train,
        model=models[model_name],
        scaler=preprocessing_config.get("scaler", "standard"),
        numeric_strategy=preprocessing_config.get("numeric_impute_strategy", "median"),
        categorical_strategy=preprocessing_config.get("categorical_impute_strategy", "most_frequent"),
    )

    pipeline.fit(X_train, y_train)

    output_dirs = create_output_dirs(ROOT / output_config.get("base_dir", "outputs"))

    builtin_table = get_builtin_feature_importance(
        model=pipeline,
        X=X_train,
    )

    if args.task == "regression":
        scoring = "neg_root_mean_squared_error"
    else:
        scoring = "accuracy"

    permutation_table = compute_permutation_importance(
        model=pipeline,
        X=X_test,
        y=y_test,
        scoring=scoring,
        n_repeats=8,
        random_state=random_state,
    )

    shap_table = None

    if not args.skip_shap:
        try:
            shap_table = compute_shap_summary(
                model=pipeline,
                X=X_test,
                max_samples=args.max_samples,
            )
        except Exception as exc:
            print("SHAP 分析未完成，流程继续运行。")
            print(f"原因: {exc}")

    explain_dir = output_dirs["reports"] / "explainability"
    figure_dir = output_dirs["figures"] / "explainability"

    save_explainability_tables(
        output_dir=explain_dir,
        builtin_table=builtin_table,
        permutation_table=permutation_table,
        shap_table=shap_table,
    )

    if builtin_table is not None and not builtin_table.empty:
        plot_importance_bar(
            table=builtin_table,
            value_column="importance",
            path=figure_dir / "builtin_feature_importance.png",
            title="模型内置特征重要性",
        )

    if permutation_table is not None and not permutation_table.empty:
        plot_importance_bar(
            table=permutation_table,
            value_column="importance_mean",
            path=figure_dir / "permutation_importance.png",
            title="Permutation Importance",
        )

    if shap_table is not None and not shap_table.empty:
        plot_importance_bar(
            table=shap_table,
            value_column="mean_abs_shap",
            path=figure_dir / "shap_importance.png",
            title="SHAP 平均绝对贡献",
        )

    print()
    print("内置特征重要性:")
    print(builtin_table.head(20))

    print()
    print("Permutation Importance:")
    print(permutation_table.head(20))

    if shap_table is not None:
        print()
        print("SHAP 重要性:")
        print(shap_table.head(20))

    print()
    print(f"输出目录: {output_dirs['base']}")


if __name__ == "__main__":
    main()