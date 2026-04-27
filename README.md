# 高温合金机器学习项目

本项目面向高温合金表格数据建模，覆盖从数据读取、数据处理、特征工程、传统机器学习、聚类、降维、深度学习、模型解释到结果报告的完整流程。

项目默认使用可复现的合成高温合金数据，便于直接运行。后续也可以替换为指定路径的 CSV 或 Excel 表格数据。

---

## 项目功能

### 数据处理

- 读取合成数据
- 读取外部 CSV 数据
- 读取外部 Excel 数据
- 检查缺失值
- 检查重复值
- 检查元素成分是否合法
- 检查目标列是否存在
- 拆分特征和目标
- 训练集、验证集、测试集划分
- 数值特征填充
- 类别特征编码
- 标准化和归一化

### 高温合金特征工程

- 元素成分归一化
- γ′ 形成元素总量
- 难熔元素总量
- 抗氧化元素总量
- Al/Ti 比值
- Ta/W 比值
- Cr/Al 比值
- Co/Ni 比值
- 平均原子量
- 平均原子半径
- 原子半径失配
- 平均电负性
- 电负性失配
- 混合熵
- 估算密度
- 估算成本指数
- 测试温度 K 转换
- 热处理参数
- 应力-温度交互项

### 传统机器学习

线性模型：

- Linear Regression
- Ridge
- Lasso
- ElasticNet
- Logistic Regression

邻近算法：

- KNN Regressor
- KNN Classifier

核方法：

- SVR
- SVM Classifier

树模型：

- Decision Tree
- Random Forest
- Extra Trees
- Gradient Boosting Tree
- XGBoost

### 聚类算法

- KMeans
- DBSCAN
- Agglomerative Clustering

### 降维算法

- PCA
- t-SNE

### 深度学习算法

基于 PyTorch：

- MLP
- CNN
- RNN
- LSTM
- Transformer

### 模型解释

- SHAP
- Permutation Importance
- Feature Importance
- SHAP summary plot
- SHAP bar plot

### 模型评估

回归任务：

- MAE
- MSE
- RMSE
- R2
- MAPE
- Max Error

分类任务：

- Accuracy
- Precision
- Recall
- F1
- Confusion Matrix

模型比较：

- 交叉验证
- 模型排行榜
- 预测值与真实值对比
- 残差分析
- 评估结果保存
- 图表生成

---

## 目录结构

```text
superalloy-ml/
│
├── README.md
├── requirements.txt
├── requirements-dl.txt
├── requirements-optional.txt
├── pyproject.toml
├── Makefile
├── .gitignore
│
├── configs/
│   └── default.yaml
│
├── data/
│   ├── README.md
│   ├── raw/
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep
│
├── scripts/
│   ├── make_dataset.py
│   ├── run_data_check.py
│   ├── run_feature_engineering.py
│   ├── run_traditional_ml.py
│   ├── run_clustering.py
│   ├── run_dimensionality_reduction.py
│   ├── run_deep_learning.py
│   ├── run_shap_analysis.py
│   ├── run_evaluation.py
│   └── run_all.py
│
├── src/
│   └── superalloy_ml/
│       ├── __init__.py
│       ├── constants.py
│       ├── utils.py
│       ├── validation.py
│       ├── data.py
│       ├── preprocessing.py
│       ├── features.py
│       ├── traditional_ml.py
│       ├── clustering.py
│       ├── dimensionality_reduction.py
│       ├── explainability.py
│       ├── evaluation.py
│       ├── reporting.py
│       └── deep_learning/
│           ├── __init__.py
│           ├── datasets.py
│           ├── mlp.py
│           ├── cnn.py
│           ├── rnn_lstm.py
│           ├── transformer.py
│           └── trainer.py
│
├── notebooks/
│   ├── 01_数据处理.ipynb
│   ├── 02_特征工程.ipynb
│   ├── 03_传统机器学习.ipynb
│   ├── 04_聚类与降维.ipynb
│   ├── 05_深度学习.ipynb
│   ├── 06_SHAP分析.ipynb
│   └── 07_模型对比.ipynb
│
├── docs/
│   ├── 项目说明.md
│   ├── 数据字段说明.md
│   ├── 特征工程说明.md
│   ├── 算法说明.md
│   ├── 评估指标说明.md
│   └── AI辅助说明.md
│
├── tests/
│   ├── test_data.py
│   ├── test_preprocessing.py
│   ├── test_features.py
│   ├── test_traditional_ml.py
│   ├── test_clustering.py
│   ├── test_dimensionality_reduction.py
│   ├── test_deep_learning.py
│   ├── test_explainability.py
│   ├── test_evaluation.py
│   └── test_reporting.py
│
└── outputs/
    ├── models/
    │   └── .gitkeep
    ├── reports/
    │   └── .gitkeep
    ├── figures/
    │   └── .gitkeep
    └── predictions/
        └── .gitkeep
```

---

## 目录说明

### `configs/`

存放项目配置文件。

主要控制：

- 数据来源
- 数据路径
- 目标列
- 测试集比例
- 随机种子
- 是否使用 XGBoost
- 是否保存模型
- 是否生成图表
- 输出目录

---

### `data/`

存放数据。

```text
data/raw/
```

用于存放原始数据，例如：

```text
data/raw/superalloy_data.csv
data/raw/superalloy_data.xlsx
```

```text
data/processed/
```

用于存放处理后的数据，例如：

```text
data/processed/synthetic_superalloy.csv
data/processed/cleaned_superalloy.csv
```

---

### `scripts/`

存放可以直接运行的脚本。

常用命令：

```bash
python scripts/make_dataset.py
python scripts/run_data_check.py
python scripts/run_feature_engineering.py
python scripts/run_traditional_ml.py
python scripts/run_clustering.py
python scripts/run_dimensionality_reduction.py
python scripts/run_deep_learning.py
python scripts/run_shap_analysis.py
python scripts/run_all.py
```

---

### `src/superalloy_ml/`

核心源码目录。

主要模块：

```text
data.py
```

负责数据生成、数据读取、数据保存、特征和目标拆分。

```text
preprocessing.py
```

负责缺失值填充、标准化、归一化、类别编码和 sklearn 预处理流水线。

```text
features.py
```

负责高温合金领域特征工程。

```text
traditional_ml.py
```

负责传统机器学习模型训练、交叉验证、模型保存和模型加载。

```text
clustering.py
```

负责聚类分析。

```text
dimensionality_reduction.py
```

负责 PCA 和 t-SNE 降维。

```text
deep_learning/
```

负责 PyTorch 模型和训练流程。

```text
explainability.py
```

负责 SHAP 和特征重要性分析。

```text
evaluation.py
```

负责回归和分类指标。

```text
reporting.py
```

负责保存表格、图像、模型结果和报告。

---

### `notebooks/`

用于交互式分析和展示，不作为主要运行入口。

项目主入口在：

```text
scripts/
```

---

### `docs/`

存放说明文档。

建议记录：

- 数据字段含义
- 特征工程逻辑
- 算法用途
- 评估指标含义
- 外部数据替换方式
- AI 辅助说明

---

### `tests/`

存放单元测试。

运行：

```bash
pytest tests/
```

---

### `outputs/`

存放运行结果。

```text
outputs/models/
```

保存模型文件。

```text
outputs/reports/
```

保存模型结果表、交叉验证结果和报告。

```text
outputs/figures/
```

保存图像。

```text
outputs/predictions/
```

保存预测结果。

---

## 快速开始

创建环境：

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

安装基础依赖：

```bash
pip install -r requirements.txt
```

生成合成数据：

```bash
python scripts/make_dataset.py
```

检查数据：

```bash
python scripts/run_data_check.py
```

运行特征工程：

```bash
python scripts/run_feature_engineering.py
```

运行传统机器学习：

```bash
python scripts/run_traditional_ml.py
```

运行测试：

```bash
pytest tests/
```

## 单独运行某个算法

本项目支持单独运行某一个算法。

### 单独运行回归模型

```bash
python scripts/run_single_model.py --task regression --model-name ridge
```

```bash
python scripts/run_single_model.py --task regression --model-name random_forest_regressor
```

```bash
python scripts/run_single_model.py --task regression --model-name xgboost_regressor
```

### 单独运行分类模型

```bash
python scripts/run_single_model.py --task classification --model-name logistic_regression
```

```bash
python scripts/run_single_model.py --task classification --model-name random_forest_classifier
```

```bash
python scripts/run_single_model.py --task classification --model-name svm_classifier
```

### 单独运行聚类算法

```bash
python scripts/run_single_model.py --task clustering --model-name kmeans
```

```bash
python scripts/run_single_model.py --task clustering --model-name dbscan
```

```bash
python scripts/run_single_model.py --task clustering --model-name agglomerative
```

### 单独运行降维算法

```bash
python scripts/run_single_model.py --task reduction --model-name pca
```

```bash
python scripts/run_single_model.py --task reduction --model-name tsne
```

### 使用外部数据单独运行

```bash
python scripts/run_single_model.py \
  --task regression \
  --model-name random_forest_regressor \
  --data-path data/raw/my_superalloy_data.csv \
  --target creep_log_life_h
```

### 保存单模型文件

```bash
python scripts/run_single_model.py \
  --task regression \
  --model-name random_forest_regressor \
  --save-model
```

---

## 使用合成数据

默认配置使用合成数据.

配置文件：

```yaml
data:
  source: synthetic
```

生成数据：

```bash
python scripts/make_dataset.py
```

生成文件：

```text
data/processed/synthetic_superalloy.csv
```

---

## 使用外部 CSV 数据

将数据放入：

```text
data/raw/my_superalloy_data.csv
```

修改配置：

```yaml
data:
  source: external
  external_path: data/raw/my_superalloy_data.csv
```

然后运行：

```bash
python scripts/run_data_check.py
python scripts/run_traditional_ml.py
```

也可以在脚本中通过参数指定数据路径：

```bash
python scripts/run_traditional_ml.py --data-path data/raw/my_superalloy_data.csv --target creep_log_life_h
```

---

## 使用外部 Excel 数据

将数据放入：

```text
data/raw/my_superalloy_data.xlsx
```

修改配置：

```yaml
data:
  source: external
  external_path: data/raw/my_superalloy_data.xlsx
```

然后运行：

```bash
python scripts/run_data_check.py
python scripts/run_traditional_ml.py
```

---

## 推荐表格字段

一行代表一个样本。

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

目标列可以不全有。只要配置中指定的目标列存在即可。

---

## 主要输出

运行完成后，结果保存在：

```text
outputs/
```

模型文件：

```text
outputs/models/
```

结果表：

```text
outputs/reports/
```

图像：

```text
outputs/figures/
```

预测结果：

```text
outputs/predictions/
```

---

## 常用命令

```bash
make install
make data
make check
make features
make traditional
make test
```

没有 `make` 的环境可以直接运行：

```bash
pip install -r requirements.txt
python scripts/make_dataset.py
python scripts/run_data_check.py
python scripts/run_feature_engineering.py
python scripts/run_traditional_ml.py
pytest tests/
```

---

## 深度学习依赖

PyTorch 相关功能单独安装：

```bash
pip install -r requirements-dl.txt
```

运行深度学习示例：

```bash
python scripts/run_deep_learning.py
```

---

## 可选依赖

SHAP、Optuna、LightGBM、CatBoost、UMAP 等功能单独安装：

```bash
pip install -r requirements-optional.txt
```

---

## 注意事项

- 合成数据只用于流程验证和代码演示。
- 外部数据建议先统一单位和字段名称。
- 元素成分默认按百分比处理。
- 如果成分总和不等于 100，可以在特征工程中自动归一化。
- 小样本数据建议重点查看交叉验证结果。
- 树模型适合表格数据，SVM 和 KNN 对标准化更敏感。
- SHAP 分析主要用于解释模型，而不是证明因果关系。