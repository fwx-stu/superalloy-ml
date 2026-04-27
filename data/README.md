# 数据目录说明

本目录用于存放项目数据。

## 目录结构

```text
data/
├── raw/
│   └── .gitkeep
└── processed/
    └── .gitkeep
```

## `data/raw/`

用于存放外部原始数据，例如：

```text
data/raw/my_superalloy_data.csv
data/raw/my_superalloy_data.xlsx
```

## `data/processed/`

用于存放处理后的数据，例如：

```text
data/processed/synthetic_superalloy.csv
data/processed/cleaned_superalloy.csv
```

## 默认合成数据

运行：

```bash
python scripts/make_dataset.py
```

会生成：

```text
data/processed/synthetic_superalloy.csv
```

## 外部数据字段建议

元素成分：

```text
Ni, Co, Cr, Al, Ti, Ta, W, Mo, Re, Nb, Hf, C, B
```

热处理参数：

```text
solution_temp_c
solution_time_h
aging_temp_c
aging_time_h
cooling_rate_c_s
```

测试条件：

```text
test_temp_c
stress_mpa
```

回归目标：

```text
yield_strength_mpa
tensile_strength_mpa
creep_log_life_h
oxidation_mass_gain_mg_cm2
```

分类目标：

```text
creep_life_class
```

数据可以只包含部分目标列。运行时指定的目标列必须存在。