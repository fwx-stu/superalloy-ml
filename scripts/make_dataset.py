from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from superalloy_ml.data import save_synthetic_dataset
from superalloy_ml.utils import get_config, print_section


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成合成高温合金数据。")
    parser.add_argument("--config", default="configs/default.yaml", help="配置文件路径。")
    parser.add_argument("--output", default=None, help="输出数据路径。")
    parser.add_argument("--n-samples", type=int, default=None, help="样本数量。")
    parser.add_argument("--random-state", type=int, default=None, help="随机种子。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_config(args.config)

    data_config = config.get("data", {})
    project_config = config.get("project", {})

    output = args.output or data_config.get(
        "synthetic_path",
        "data/processed/synthetic_superalloy.csv",
    )
    n_samples = args.n_samples or int(data_config.get("n_samples", 1200))
    random_state = args.random_state or int(project_config.get("random_state", 42))

    print_section("生成合成高温合金数据")

    df = save_synthetic_dataset(
        path=output,
        n_samples=n_samples,
        random_state=random_state,
    )

    print(f"输出路径: {ROOT / output if not Path(output).is_absolute() else output}")
    print(f"数据形状: {df.shape}")
    print()
    print(df.head())


if __name__ == "__main__":
    main()