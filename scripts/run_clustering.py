from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from superalloy_ml.constants import TARGET_COLUMNS
from superalloy_ml.clustering import (
    make_cluster_assignment_table,
    make_cluster_embedding_table,
    run_all_clustering,
)
from superalloy_ml.data import load_configured_dataset, read_table
from superalloy_ml.features import get_numeric_feature_columns, make_features
from superalloy_ml.reporting import create_output_dirs, save_table
from superalloy_ml.utils import get_config, print_section


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行聚类分析。")
    parser.add_argument("--config", default="configs/default.yaml", help="配置文件路径。")
    parser.add_argument("--data-path", default=None, help="外部数据路径，支持 CSV 或 Excel。")
    parser.add_argument("--n-clusters", type=int, default=3, help="KMeans 和层次聚类的簇数量。")
    parser.add_argument("--dbscan-eps", type=float, default=1.5, help="DBSCAN 的 eps 参数。")
    parser.add_argument("--dbscan-min-samples", type=int, default=8, help="DBSCAN 的 min_samples 参数。")
    return parser.parse_args()


def plot_cluster_embedding(table, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.scatter(table["x"], table["y"], c=table["cluster"], alpha=0.75)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    config = get_config(args.config)

    print_section("运行聚类分析")

    if args.data_path:
        df = read_table(args.data_path)
    else:
        df = load_configured_dataset(config)

    feature_config = config.get("features", {})
    preprocessing_config = config.get("preprocessing", {})
    project_config = config.get("project", {})
    output_base_dir = config.get("outputs", {}).get("base_dir", "outputs")

    df = make_features(
        df,
        normalize=bool(feature_config.get("normalize_composition", True)),
        add_composition=bool(feature_config.get("add_composition_features", True)),
        add_physical=bool(feature_config.get("add_physical_descriptors", True)),
        add_processing=bool(feature_config.get("add_processing_features", True)),
        add_interactions=bool(feature_config.get("add_interaction_features", True)),
    )

    feature_columns = get_numeric_feature_columns(df, TARGET_COLUMNS)
    X = df[feature_columns]

    output_dirs = create_output_dirs(ROOT / output_base_dir)

    summary, results, _ = run_all_clustering(
        X=X,
        n_clusters=args.n_clusters,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
        scaler=preprocessing_config.get("scaler", "standard"),
        random_state=int(project_config.get("random_state", 42)),
    )

    assignment_table = make_cluster_assignment_table(results, index=df.index)

    save_table(summary, output_dirs["reports"] / "clustering_summary.csv")
    save_table(assignment_table, output_dirs["predictions"] / "clustering_assignments.csv")

    print("聚类结果摘要:")
    print(summary)

    for name, result in results.items():
        embedding_table = make_cluster_embedding_table(result)
        save_table(
            embedding_table,
            output_dirs["predictions"] / f"{name}_cluster_embedding.csv",
        )
        plot_cluster_embedding(
            embedding_table,
            output_dirs["figures"] / f"{name}_cluster_embedding.png",
            title=f"{name} 聚类结果",
        )

    print()
    print(f"结果已保存到: {output_dirs['base']}")


if __name__ == "__main__":
    main()