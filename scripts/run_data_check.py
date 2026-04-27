from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from superalloy_ml.data import load_configured_dataset, read_table
from superalloy_ml.utils import get_config, print_section
from superalloy_ml.validation import check_dataset_ready, summarize_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查高温合金表格数据。")
    parser.add_argument("--config", default="configs/default.yaml", help="配置文件路径。")
    parser.add_argument("--data-path", default=None, help="外部数据路径，支持 CSV 或 Excel。")
    parser.add_argument("--target", default=None, help="目标列名称。")
    parser.add_argument(
        "--allow-composition-not-100",
        action="store_true",
        help="允许元素成分总和不等于 100。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_config(args.config)

    print_section("检查数据")

    if args.data_path:
        df = read_table(args.data_path)
    else:
        df = load_configured_dataset(config)

    target = args.target
    if target is None:
        target = config.get("target", {}).get("regression")

    check_dataset_ready(
        df,
        target=target,
        require_composition_sum_100=not args.allow_composition_not_100,
    )

    summary = summarize_dataset(df)

    print("数据检查通过。")
    print()
    print("数据摘要:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print()
    print("字段列表:")
    for column in df.columns:
        print(f"  - {column}")


if __name__ == "__main__":
    main()