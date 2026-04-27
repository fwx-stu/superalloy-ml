from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from superalloy_ml.constants import TARGET_COLUMNS
from superalloy_ml.data import load_configured_dataset, read_table, save_processed_dataset
from superalloy_ml.features import get_numeric_feature_columns, make_features
from superalloy_ml.utils import get_config, print_section


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行高温合金特征工程。")
    parser.add_argument("--config", default="configs/default.yaml", help="配置文件路径。")
    parser.add_argument("--data-path", default=None, help="外部数据路径，支持 CSV 或 Excel。")
    parser.add_argument(
        "--output",
        default="data/processed/featured_superalloy.csv",
        help="特征工程结果保存路径。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_config(args.config)

    print_section("运行特征工程")

    if args.data_path:
        df = read_table(args.data_path)
    else:
        df = load_configured_dataset(config)

    feature_config = config.get("features", {})

    featured_df = make_features(
        df,
        normalize=bool(feature_config.get("normalize_composition", True)),
        add_composition=bool(feature_config.get("add_composition_features", True)),
        add_physical=bool(feature_config.get("add_physical_descriptors", True)),
        add_processing=bool(feature_config.get("add_processing_features", True)),
        add_interactions=bool(feature_config.get("add_interaction_features", True)),
    )

    save_processed_dataset(featured_df, args.output)

    feature_columns = get_numeric_feature_columns(featured_df, TARGET_COLUMNS)

    print(f"原始数据形状: {df.shape}")
    print(f"特征工程后数据形状: {featured_df.shape}")
    print(f"可用于建模的数值特征数量: {len(feature_columns)}")
    print(f"保存路径: {args.output}")

    print()
    print("部分特征列:")
    for column in feature_columns[:40]:
        print(f"  - {column}")


if __name__ == "__main__":
    main()