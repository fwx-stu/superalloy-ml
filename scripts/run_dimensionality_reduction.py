from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from superalloy_ml.constants import TARGET_COLUMNS
from superalloy_ml.data import load_configured_dataset, read_table
from superalloy_ml.dimensionality_reduction import make_embedding_table, run_all_reductions
from superalloy_ml.features import get_numeric_feature_columns, make_features
from superalloy_ml.reporting import create_output_dirs, save_table
from superalloy_ml.utils import get_config, print_section


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 PCA 和 t-SNE 降维。")
    parser.add_argument("--config", default="configs/default.yaml", help="配置文件路径。")
    parser.add_argument("--data-path", default=None, help="外部数据路径，支持 CSV 或 Excel。")
    parser.add_argument("--label-column", default=None, help="可选标签列，用于图像着色。")
    parser.add_argument("--tsne-perplexity", type=float, default=30.0, help="t-SNE perplexity 参数。")
    return parser.parse_args()


def plot_embedding(table, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))

    if "label" in table.columns:
        plt.scatter(table["x"], table["y"], c=table["label"], alpha=0.75)
    else:
        plt.scatter(table["x"], table["y"], alpha=0.75)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    config = get_config(args.config)

    print_section("运行降维分析")

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

    label = None
    if args.label_column and args.label_column in df.columns:
        label = df[args.label_column].to_numpy()
    elif "creep_life_class" in df.columns:
        label = df["creep_life_class"].to_numpy()

    output_dirs = create_output_dirs(ROOT / output_base_dir)

    summary, results, _ = run_all_reductions(
        X=X,
        scaler=preprocessing_config.get("scaler", "standard"),
        random_state=int(project_config.get("random_state", 42)),
        tsne_perplexity=args.tsne_perplexity,
    )

    save_table(summary, output_dirs["reports"] / "dimensionality_reduction_summary.csv")

    print("降维结果摘要:")
    print(summary)

    for name, result in results.items():
        embedding_table = make_embedding_table(result, label=label)

        save_table(
            embedding_table,
            output_dirs["predictions"] / f"{name}_embedding.csv",
        )

        plot_embedding(
            embedding_table,
            output_dirs["figures"] / f"{name}_embedding.png",
            title=f"{name} 二维降维结果",
        )

    print()
    print(f"结果已保存到: {output_dirs['base']}")


if __name__ == "__main__":
    main()