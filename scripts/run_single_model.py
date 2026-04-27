from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from superalloy_ml.clustering import (
    make_cluster_embedding_table,
    prepare_clustering_matrix,
    run_agglomerative,
    run_dbscan,
    run_kmeans,
)
from superalloy_ml.constants import TARGET_COLUMNS
from superalloy_ml.data import load_configured_dataset, read_table, split_features_targets
from superalloy_ml.dimensionality_reduction import (
    make_embedding_table,
    prepare_reduction_matrix,
    run_pca,
    run_tsne,
)
from superalloy_ml.evaluation import classification_metrics, regression_metrics
from superalloy_ml.features import get_numeric_feature_columns, make_features
from superalloy_ml.reporting import (
    create_output_dirs,
    plot_confusion_matrix,
    plot_prediction_vs_actual,
    plot_residuals,
    save_table,
)
from superalloy_ml.traditional_ml import build_pipeline, get_models, save_model
from superalloy_ml.utils import ensure_dir, get_config, print_section


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="单独运行一个机器学习算法。")

    parser.add_argument(
        "--task",
        required=True,
        choices=["regression", "classification", "clustering", "reduction"],
        help="任务类型。",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="模型名称。",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="配置文件路径。",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="外部数据路径，支持 CSV 或 Excel。",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="目标列名称。回归或分类任务需要。",
    )
    parser.add_argument(
        "--no-xgboost",
        action="store_true",
        help="不使用 XGBoost。",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=3,
        help="KMeans 和层次聚类的簇数量。",
    )
    parser.add_argument(
        "--dbscan-eps",
        type=float,
        default=1.5,
        help="DBSCAN 的 eps 参数。",
    )
    parser.add_argument(
        "--dbscan-min-samples",
        type=int,
        default=8,
        help="DBSCAN 的 min_samples 参数。",
    )
    parser.add_argument(
        "--tsne-perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity 参数。",
    )
    parser.add_argument(
        "--label-column",
        default=None,
        help="降维可视化标签列。",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="保存模型。",
    )

    return parser.parse_args()


def load_and_prepare_dataframe(args: argparse.Namespace, config: dict) -> pd.DataFrame:
    if args.data_path:
        df = read_table(args.data_path)
    else:
        df = load_configured_dataset(config)

    feature_config = config.get("features", {})

    df = make_features(
        df,
        normalize=bool(feature_config.get("normalize_composition", True)),
        add_composition=bool(feature_config.get("add_composition_features", True)),
        add_physical=bool(feature_config.get("add_physical_descriptors", True)),
        add_processing=bool(feature_config.get("add_processing_features", True)),
        add_interactions=bool(feature_config.get("add_interaction_features", True)),
    )

    return df


def resolve_target(args: argparse.Namespace, config: dict) -> str:
    if args.target:
        return args.target

    if args.task == "regression":
        return config.get("target", {}).get("regression", "creep_log_life_h")

    if args.task == "classification":
        return config.get("target", {}).get("classification", "creep_life_class")

    return ""


def run_single_supervised_model(
    args: argparse.Namespace,
    config: dict,
    df: pd.DataFrame,
    output_dirs: dict[str, Path],
) -> None:
    task = args.task
    target = resolve_target(args, config)

    if target not in df.columns:
        raise ValueError(f"目标列不存在: {target}")

    include_xgboost = bool(config.get("modeling", {}).get("use_xgboost", True))
    include_xgboost = include_xgboost and not args.no_xgboost

    random_state = int(config.get("project", {}).get("random_state", 42))
    test_size = float(config.get("data", {}).get("test_size", 0.2))

    preprocessing_config = config.get("preprocessing", {})
    scaler = preprocessing_config.get("scaler", "standard")
    numeric_strategy = preprocessing_config.get("numeric_impute_strategy", "median")
    categorical_strategy = preprocessing_config.get(
        "categorical_impute_strategy",
        "most_frequent",
    )

    X, y = split_features_targets(df, target=target)

    models = get_models(
        task=task,
        random_state=random_state,
        include_xgboost=include_xgboost,
    )

    if args.model_name not in models:
        available = "\n".join([f"  - {name}" for name in models])
        raise ValueError(f"模型不存在: {args.model_name}\n可选模型:\n{available}")

    stratify = y if task == "classification" else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    pipeline = build_pipeline(
        X_train,
        model=models[args.model_name],
        scaler=scaler,
        numeric_strategy=numeric_strategy,
        categorical_strategy=categorical_strategy,
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    if task == "regression":
        metrics = regression_metrics(y_test, y_pred)

        prediction_df = pd.DataFrame(
            {
                "y_true": y_test.to_numpy(),
                "y_pred": y_pred,
                "residual": y_test.to_numpy() - y_pred,
            }
        )

        plot_prediction_vs_actual(
            y_true=y_test.to_numpy(),
            y_pred=y_pred,
            path=output_dirs["figures"] / f"single_{args.model_name}_prediction_vs_actual.png",
            title=f"{args.model_name} 预测值与真实值对比",
        )

        plot_residuals(
            y_true=y_test.to_numpy(),
            y_pred=y_pred,
            path=output_dirs["figures"] / f"single_{args.model_name}_residuals.png",
            title=f"{args.model_name} 残差图",
        )

    else:
        metrics = classification_metrics(y_test, y_pred)

        prediction_df = pd.DataFrame(
            {
                "y_true": y_test.to_numpy(),
                "y_pred": y_pred,
            }
        )

        plot_confusion_matrix(
            y_true=y_test.to_numpy(),
            y_pred=y_pred,
            path=output_dirs["figures"] / f"single_{args.model_name}_confusion_matrix.png",
            title=f"{args.model_name} 混淆矩阵",
        )

    result_df = pd.DataFrame(
        [
            {
                "task": task,
                "model_name": args.model_name,
                "target": target,
                "train_size": len(X_train),
                "test_size": len(X_test),
                **metrics,
            }
        ]
    )

    save_table(
        result_df,
        output_dirs["reports"] / f"single_{task}_{args.model_name}_metrics.csv",
    )
    save_table(
        prediction_df,
        output_dirs["predictions"] / f"single_{task}_{args.model_name}_predictions.csv",
    )

    if args.save_model:
        save_model(
            pipeline,
            output_dirs["models"] / f"single_{task}_{args.model_name}.joblib",
        )

    print("评估结果:")
    print(result_df)


def run_single_clustering_model(
    args: argparse.Namespace,
    config: dict,
    df: pd.DataFrame,
    output_dirs: dict[str, Path],
) -> None:
    random_state = int(config.get("project", {}).get("random_state", 42))
    scaler = config.get("preprocessing", {}).get("scaler", "standard")

    feature_columns = get_numeric_feature_columns(df, TARGET_COLUMNS)
    X = df[feature_columns]

    X_processed, preprocessor = prepare_clustering_matrix(
        X,
        scaler=scaler,
    )

    if args.model_name == "kmeans":
        result = run_kmeans(
            X_processed,
            n_clusters=args.n_clusters,
            random_state=random_state,
        )
    elif args.model_name == "dbscan":
        result = run_dbscan(
            X_processed,
            eps=args.dbscan_eps,
            min_samples=args.dbscan_min_samples,
            random_state=random_state,
        )
    elif args.model_name in {"agglomerative", "agglomerative_clustering"}:
        result = run_agglomerative(
            X_processed,
            n_clusters=args.n_clusters,
            random_state=random_state,
        )
    else:
        raise ValueError(
            "聚类模型名称不支持。可选: kmeans, dbscan, agglomerative"
        )

    metrics_df = pd.DataFrame(
        [
            {
                "task": "clustering",
                "model_name": args.model_name,
                **result.metrics,
            }
        ]
    )

    embedding_df = make_cluster_embedding_table(result)

    save_table(
        metrics_df,
        output_dirs["reports"] / f"single_clustering_{args.model_name}_metrics.csv",
    )
    save_table(
        embedding_df,
        output_dirs["predictions"] / f"single_clustering_{args.model_name}_embedding.csv",
    )

    assignment_df = pd.DataFrame(
        {
            "cluster": result.labels,
        }
    )
    save_table(
        assignment_df,
        output_dirs["predictions"] / f"single_clustering_{args.model_name}_labels.csv",
    )

    if args.save_model:
        model_dir = ensure_dir(output_dirs["models"])
        joblib.dump(
            {
                "model": result.model,
                "preprocessor": preprocessor,
                "metrics": result.metrics,
            },
            model_dir / f"single_clustering_{args.model_name}.joblib",
        )

    print("聚类结果:")
    print(metrics_df)


def run_single_reduction_model(
    args: argparse.Namespace,
    config: dict,
    df: pd.DataFrame,
    output_dirs: dict[str, Path],
) -> None:
    random_state = int(config.get("project", {}).get("random_state", 42))
    scaler = config.get("preprocessing", {}).get("scaler", "standard")

    feature_columns = get_numeric_feature_columns(df, TARGET_COLUMNS)
    X = df[feature_columns]

    X_processed, preprocessor = prepare_reduction_matrix(
        X,
        scaler=scaler,
    )

    if args.model_name == "pca":
        result = run_pca(
            X_processed,
            n_components=2,
            random_state=random_state,
        )
    elif args.model_name in {"tsne", "t-sne"}:
        result = run_tsne(
            X_processed,
            n_components=2,
            perplexity=args.tsne_perplexity,
            random_state=random_state,
        )
    else:
        raise ValueError("降维模型名称不支持。可选: pca, tsne")

    label = None
    if args.label_column and args.label_column in df.columns:
        label = df[args.label_column].to_numpy()
    elif "creep_life_class" in df.columns:
        label = df["creep_life_class"].to_numpy()

    embedding_df = make_embedding_table(result, label=label)

    metrics_df = pd.DataFrame(
        [
            {
                "task": "reduction",
                "model_name": args.model_name,
                **result.metadata,
            }
        ]
    )

    save_table(
        metrics_df,
        output_dirs["reports"] / f"single_reduction_{args.model_name}_metrics.csv",
    )
    save_table(
        embedding_df,
        output_dirs["predictions"] / f"single_reduction_{args.model_name}_embedding.csv",
    )

    if args.save_model:
        model_dir = ensure_dir(output_dirs["models"])
        joblib.dump(
            {
                "model": result.model,
                "preprocessor": preprocessor,
                "metadata": result.metadata,
            },
            model_dir / f"single_reduction_{args.model_name}.joblib",
        )

    print("降维结果:")
    print(metrics_df)


def main() -> None:
    args = parse_args()
    config = get_config(args.config)

    print_section("单模型运行")

    df = load_and_prepare_dataframe(args, config)

    output_base_dir = config.get("outputs", {}).get("base_dir", "outputs")
    output_dirs = create_output_dirs(ROOT / output_base_dir)

    print(f"任务类型: {args.task}")
    print(f"模型名称: {args.model_name}")

    if args.task in {"regression", "classification"}:
        run_single_supervised_model(
            args=args,
            config=config,
            df=df,
            output_dirs=output_dirs,
        )
    elif args.task == "clustering":
        run_single_clustering_model(
            args=args,
            config=config,
            df=df,
            output_dirs=output_dirs,
        )
    elif args.task == "reduction":
        run_single_reduction_model(
            args=args,
            config=config,
            df=df,
            output_dirs=output_dirs,
        )
    else:
        raise ValueError(f"未知任务类型: {args.task}")

    print()
    print(f"输出目录: {output_dirs['base']}")


if __name__ == "__main__":
    main()