# 高温合金机器学习项目

本项目面向高温合金表格数据建模，覆盖从数据读取、数据处理、特征工程、传统机器学习、聚类、降维、深度学习、模型解释到结果报告的完整流程。

项目默认使用可复现的合成高温合金数据，便于直接运行。后续也可以替换为指定路径的 CSV 或 Excel 表格数据。

---

## 项目可以完成什么任务

本项目主要支持以下任务：

| 任务 | 对应脚本 | 说明 |
|---|---|---|
| 生成合成数据 | `scripts/make_dataset.py` | 生成可直接运行的高温合金表格数据 |
| 数据检查 | `scripts/run_data_check.py` | 检查缺失值、重复值、元素成分、目标列 |
| 特征工程 | `scripts/run_feature_engineering.py` | 生成高温合金领域特征 |
| 传统机器学习 | `scripts/run_traditional_ml.py` | 一次运行多种传统机器学习模型 |
| 单个算法运行 | `scripts/run_single_model.py` | 单独运行某一个模型或算法 |
| 聚类分析 | `scripts/run_clustering.py` | 运行 KMeans、DBSCAN、层次聚类 |
| 降维分析 | `scripts/run_dimensionality_reduction.py` | 运行 PCA 和 t-SNE |
| 深度学习 | `scripts/run_deep_learning.py` | 运行 PyTorch MLP、CNN、RNN、LSTM、Transformer |
| 模型解释 | `scripts/run_shap_analysis.py` | 生成特征重要性、Permutation Importance、SHAP 图 |
| 结果汇总 | `scripts/run_evaluation.py` | 汇总模型结果并生成报告 |
| 自动全流程 | `scripts/run_all.py` | 从数据生成到结果汇总一键运行 |

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
- 训练集、测试集划分
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

- Feature Importance
- Permutation Importance
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
│   ├── run_single_model.py
│   ├── run_clustering.py
│   ├── run_dimensionality_reduction.py
│   ├── run_deep_learning.py
│   ├── run_shap_analysis.py
│   ├── run_evaluation.py
│   ├── run_all.py
│   └── check_project_structure.py
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
│   └── README.md
│
├── docs/
│   ├── 项目说明.md
│   ├── 数据字段说明.md
│   ├── 特征工程说明.md
│   ├── 算法说明.md
│   ├── 评估指标说明.md
│   ├── 常见问题.md
│   ├── 运行说明.md
│   └── AI辅助说明.md
│
├── tests/
│   ├── test_data.py
│   ├── test_validation.py
│   ├── test_preprocessing.py
│   ├── test_features.py
│   ├── test_traditional_ml.py
│   ├── test_clustering.py
│   ├── test_dimensionality_reduction.py
│   ├── test_deep_learning.py
│   ├── test_explainability.py
│   ├── test_evaluation.py
│   ├── test_reporting.py
│   ├── test_run_single_model.py
│   └── test_project_structure.py
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

主要文件：

```text
configs/default.yaml
```

---

### `data/`

存放数据。

```text
data/raw/
```

用于存放外部原始数据，例如：

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
data/processed/featured_superalloy.csv
```

---

### `scripts/`

存放可以直接运行的脚本，是项目主要入口。

常用脚本：

```text
make_dataset.py                    生成合成数据
run_data_check.py                  检查数据
run_feature_engineering.py         运行特征工程
run_traditional_ml.py              运行传统机器学习模型集合
run_single_model.py                单独运行某个算法
run_clustering.py                  运行聚类分析
run_dimensionality_reduction.py    运行 PCA 和 t-SNE
run_deep_learning.py               运行 PyTorch 深度学习模型
run_shap_analysis.py               运行模型解释分析
run_evaluation.py                  汇总结果
run_all.py                         自动运行完整流程
check_project_structure.py         检查项目结构是否完整
```

---

### `src/superalloy_ml/`

核心源码目录。

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

负责 KMeans、DBSCAN、层次聚类。

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

负责 SHAP、Permutation Importance 和特征重要性分析。

```text
evaluation.py
```

负责回归和分类指标。

```text
reporting.py
```

负责保存表格、图像、预测结果和报告。

---

### `docs/`

存放说明文档。

```text
项目说明.md          项目整体流程说明
数据字段说明.md      数据列含义和字段要求
特征工程说明.md      高温合金特征工程逻辑
算法说明.md          项目中包含的算法说明
评估指标说明.md      回归和分类指标说明
常见问题.md          数据替换、依赖、输出等问题
运行说明.md          常用运行命令
AI辅助说明.md        AI 辅助使用边界
```

---

### `tests/`

存放自动测试代码。

运行：

```bash
pytest tests/
```

用于检查：

- 数据生成是否正常
- 数据读取是否正常
- 特征工程是否正常
- 模型训练是否正常
- 聚类和降维是否正常
- 图像和报告是否能生成
- 项目结构是否完整

---

### `outputs/`

存放运行结果。

```text
outputs/models/       保存模型文件
outputs/reports/      保存结果表和报告
outputs/figures/      保存图像
outputs/predictions/  保存预测结果
```

---

## 环境准备

### 创建虚拟环境

Linux / macOS：

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 安装基础依赖

```bash
pip install -r requirements.txt
```

### 安装深度学习依赖

如果需要运行 PyTorch 模型：

```bash
pip install -r requirements-dl.txt
```

### 安装可选依赖

如果需要完整 SHAP、Optuna、LightGBM、CatBoost、UMAP 等功能：

```bash
pip install -r requirements-optional.txt
```

---

## 推荐运行方式

### 方式一：自动运行完整流程

这是最推荐的方式。会自动完成：

```text
生成合成数据
数据检查
特征工程
传统机器学习
聚类分析
降维分析
模型解释
结果汇总
```

轻量版：

```bash
python scripts/run_all.py --skip-shap --no-xgboost
```

说明：

```text
--skip-shap      跳过 SHAP，避免可选依赖问题
--no-xgboost     不运行 XGBoost，速度更快
```

完整基础版：

```bash
python scripts/run_all.py --skip-shap
```

包含 SHAP 图：

```bash
python scripts/run_all.py
```

如果要包含深度学习：

```bash
python scripts/run_all.py --include-deep-learning --skip-shap
```

如果已经安装 SHAP 和 PyTorch，也可以运行：

```bash
python scripts/run_all.py --include-deep-learning
```

---

### 方式二：逐步运行完整流程

适合第一次检查项目是否正常。

```bash
python scripts/make_dataset.py
python scripts/run_data_check.py
python scripts/run_feature_engineering.py
python scripts/run_traditional_ml.py --no-xgboost
python scripts/run_clustering.py
python scripts/run_dimensionality_reduction.py
python scripts/run_shap_analysis.py --skip-shap
python scripts/run_evaluation.py
```

如果每一步都正常，最后运行：

```bash
pytest tests/
```

---

### 方式三：只运行某一个任务

#### 只生成数据

```bash
python scripts/make_dataset.py
```

输出：

```text
data/processed/synthetic_superalloy.csv
```

#### 只检查数据

```bash
python scripts/run_data_check.py
```

如果元素成分总和不等于 100，但你希望暂时跳过该检查：

```bash
python scripts/run_data_check.py --allow-composition-not-100
```

#### 只运行特征工程

```bash
python scripts/run_feature_engineering.py
```

输出：

```text
data/processed/featured_superalloy.csv
```

#### 只运行传统机器学习

```bash
python scripts/run_traditional_ml.py --no-xgboost
```

包含 XGBoost：

```bash
python scripts/run_traditional_ml.py
```

只跑回归任务：

```bash
python scripts/run_traditional_ml.py --skip-classification
```

只跑分类任务：

```bash
python scripts/run_traditional_ml.py --skip-regression
```

指定回归目标：

```bash
python scripts/run_traditional_ml.py --regression-target creep_log_life_h
```

指定分类目标：

```bash
python scripts/run_traditional_ml.py --classification-target creep_life_class
```

#### 只运行聚类

```bash
python scripts/run_clustering.py
```

指定簇数量：

```bash
python scripts/run_clustering.py --n-clusters 4
```

调整 DBSCAN 参数：

```bash
python scripts/run_clustering.py --dbscan-eps 2.0 --dbscan-min-samples 5
```

#### 只运行降维

```bash
python scripts/run_dimensionality_reduction.py
```

指定 t-SNE 参数：

```bash
python scripts/run_dimensionality_reduction.py --tsne-perplexity 20
```

指定着色标签列：

```bash
python scripts/run_dimensionality_reduction.py --label-column creep_life_class
```

#### 只运行解释性分析

跳过 SHAP：

```bash
python scripts/run_shap_analysis.py --skip-shap
```

生成 SHAP 图：

```bash
python scripts/run_shap_analysis.py
```

指定解释模型：

```bash
python scripts/run_shap_analysis.py --model-name random_forest_regressor
```

指定解释目标：

```bash
python scripts/run_shap_analysis.py --target creep_log_life_h
```

分类解释：

```bash
python scripts/run_shap_analysis.py --task classification --target creep_life_class
```

#### 只运行结果汇总

```bash
python scripts/run_evaluation.py
```

输出：

```text
outputs/reports/summary.json
outputs/reports/summary_report.md
```

---

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
python scripts/run_single_model.py --task regression --model-name decision_tree_regressor
```

```bash
python scripts/run_single_model.py --task regression --model-name gradient_boosting_regressor
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
python scripts/run_single_model.py --task classification --model-name decision_tree_classifier
```

```bash
python scripts/run_single_model.py --task classification --model-name svm_classifier
```

```bash
python scripts/run_single_model.py --task classification --model-name xgboost_classifier
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

### 保存单模型文件

```bash
python scripts/run_single_model.py \
  --task regression \
  --model-name random_forest_regressor \
  --save-model
```

输出：

```text
outputs/models/single_regression_random_forest_regressor.joblib
```

---

## 如何修改算法参数

机器学习建模时经常需要调整算法参数。本项目支持两种调参方式：

```text
1. 直接在代码中修改默认参数
2. 通过命令行参数修改部分算法参数
```

后续也可以扩展为从 `configs/default.yaml` 统一读取模型参数。

---

### 传统机器学习参数在哪里修改

传统机器学习模型的参数主要在：

```text
src/superalloy_ml/traditional_ml.py
```

重点看两个函数：

```python
get_regression_models()
get_classification_models()
```

其中：

```python
get_regression_models()
```

负责回归模型，例如：

```text
Linear Regression
Ridge
Lasso
ElasticNet
KNN Regressor
SVR
Decision Tree Regressor
Random Forest Regressor
Extra Trees Regressor
Gradient Boosting Regressor
XGBoost Regressor
```

```python
get_classification_models()
```

负责分类模型，例如：

```text
Logistic Regression
KNN Classifier
SVM Classifier
Decision Tree Classifier
Random Forest Classifier
Extra Trees Classifier
Gradient Boosting Classifier
XGBoost Classifier
```

---

### 示例 1：修改随机森林参数

在 `src/superalloy_ml/traditional_ml.py` 中找到：

```python
"random_forest_regressor": RandomForestRegressor(
    n_estimators=160,
    max_depth=None,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=random_state,
),
```

可以改成：

```python
"random_forest_regressor": RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=3,
    max_features="sqrt",
    n_jobs=-1,
    random_state=random_state,
),
```

常见参数含义：

```text
n_estimators      树的数量，越大通常越稳定，但训练更慢
max_depth         单棵树最大深度，用于控制过拟合
min_samples_leaf  叶子节点最少样本数，越大模型越平滑
max_features      每次分裂使用的特征数量
random_state      随机种子，保证结果可复现
```

---

### 示例 2：修改 XGBoost 参数

在 `src/superalloy_ml/traditional_ml.py` 中找到：

```python
"xgboost_regressor": XGBRegressor(
    n_estimators=180,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="reg:squarederror",
    random_state=random_state,
    n_jobs=1,
    verbosity=0,
),
```

可以改成：

```python
"xgboost_regressor": XGBRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=5,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=1.0,
    reg_alpha=0.0,
    objective="reg:squarederror",
    random_state=random_state,
    n_jobs=1,
    verbosity=0,
),
```

常见参数含义：

```text
n_estimators       树的数量
learning_rate      学习率，越小通常需要更多树
max_depth          单棵树最大深度
subsample          每棵树使用的样本比例
colsample_bytree   每棵树使用的特征比例
reg_lambda         L2 正则化
reg_alpha          L1 正则化
```

---

### 示例 3：修改 SVR 参数

在 `src/superalloy_ml/traditional_ml.py` 中找到：

```python
"svr": SVR(kernel="rbf", C=10.0, epsilon=0.05),
```

可以改成：

```python
"svr": SVR(
    kernel="rbf",
    C=50.0,
    gamma="scale",
    epsilon=0.02,
),
```

常见参数含义：

```text
kernel    核函数，常用 rbf、linear、poly
C         惩罚系数，越大越重视训练误差
gamma     RBF 核的影响范围
epsilon   回归误差容忍区间
```

---

### 示例 4：修改 KNN 参数

在 `src/superalloy_ml/traditional_ml.py` 中找到：

```python
"knn_regressor": KNeighborsRegressor(n_neighbors=7),
```

可以改成：

```python
"knn_regressor": KNeighborsRegressor(
    n_neighbors=5,
    weights="distance",
    metric="minkowski",
),
```

常见参数含义：

```text
n_neighbors   邻居数量
weights       uniform 表示等权重，distance 表示距离越近权重越大
metric        距离度量方式
```

---

### 示例 5：修改逻辑回归参数

在 `src/superalloy_ml/traditional_ml.py` 中找到：

```python
"logistic_regression": LogisticRegression(
    max_iter=3000,
    class_weight="balanced",
    random_state=random_state,
),
```

可以改成：

```python
"logistic_regression": LogisticRegression(
    max_iter=5000,
    C=0.5,
    class_weight="balanced",
    solver="lbfgs",
    random_state=random_state,
),
```

常见参数含义：

```text
max_iter       最大迭代次数
C              正则化强度的倒数，越小正则越强
class_weight   类别权重，balanced 适合类别不平衡
solver         优化器
```

---

### 聚类参数如何修改

聚类脚本已经支持命令行传参。

修改 KMeans 或层次聚类的簇数量：

```bash
python scripts/run_clustering.py --n-clusters 4
```

修改 DBSCAN 参数：

```bash
python scripts/run_clustering.py --dbscan-eps 2.0 --dbscan-min-samples 5
```

单独运行 KMeans 并指定簇数量：

```bash
python scripts/run_single_model.py \
  --task clustering \
  --model-name kmeans \
  --n-clusters 4
```

单独运行 DBSCAN 并指定参数：

```bash
python scripts/run_single_model.py \
  --task clustering \
  --model-name dbscan \
  --dbscan-eps 2.0 \
  --dbscan-min-samples 5
```

参数含义：

```text
n-clusters          KMeans 和层次聚类的簇数量
dbscan-eps          DBSCAN 邻域半径
dbscan-min-samples  DBSCAN 形成核心点所需的最小样本数
```

---

### 降维参数如何修改

修改 t-SNE perplexity：

```bash
python scripts/run_dimensionality_reduction.py --tsne-perplexity 20
```

单独运行 t-SNE：

```bash
python scripts/run_single_model.py \
  --task reduction \
  --model-name tsne \
  --tsne-perplexity 20
```

参数含义：

```text
tsne-perplexity   t-SNE 的邻域复杂度，通常可尝试 5、10、20、30、50
```

PCA 默认降到二维。如果需要修改 PCA 维度，可以在：

```text
src/superalloy_ml/dimensionality_reduction.py
```

中修改：

```python
run_pca(X_processed, n_components=2)
```

---

### 深度学习参数如何修改

深度学习脚本支持命令行修改训练轮数和批大小：

```bash
python scripts/run_deep_learning.py --epochs 20 --batch-size 32
```

参数含义：

```text
epochs      训练轮数
batch-size  每次训练使用的样本数量
```

模型结构参数在以下文件中修改：

```text
src/superalloy_ml/deep_learning/mlp.py
src/superalloy_ml/deep_learning/cnn.py
src/superalloy_ml/deep_learning/rnn_lstm.py
src/superalloy_ml/deep_learning/transformer.py
```

例如 MLP 隐藏层默认在 `run_deep_learning.py` 中设置：

```python
model = TabularMLP(
    input_dim=bundle.input_dim,
    output_dim=1,
    hidden_dims=[128, 64],
    dropout=0.1,
)
```

可以改成：

```python
model = TabularMLP(
    input_dim=bundle.input_dim,
    output_dim=1,
    hidden_dims=[256, 128, 64],
    dropout=0.2,
)
```

参数含义：

```text
hidden_dims  隐藏层神经元数量
dropout      随机失活比例，用于缓解过拟合
```

---

### SHAP 参数如何修改

修改 SHAP 使用的最大样本数：

```bash
python scripts/run_shap_analysis.py --max-samples 300
```

如果只想运行内置特征重要性和 Permutation Importance：

```bash
python scripts/run_shap_analysis.py --skip-shap
```

指定解释某个模型：

```bash
python scripts/run_shap_analysis.py --model-name random_forest_regressor
```

指定解释某个目标：

```bash
python scripts/run_shap_analysis.py --target creep_log_life_h
```

参数含义：

```text
max-samples  用于 SHAP 计算的样本数量，越大越慢
model-name   指定要解释的模型
target       指定要解释的目标列
skip-shap    跳过 SHAP，只输出其他解释方法
```

---

### 推荐调参顺序

对于材料表格数据，建议优先尝试树模型和梯度提升模型。

推荐顺序：

```text
1. Random Forest
2. Extra Trees
3. Gradient Boosting
4. XGBoost
5. SVR
6. Ridge / Lasso / ElasticNet
7. KNN
```

小样本数据建议重点关注：

```text
交叉验证均值
交叉验证标准差
测试集误差
残差图
特征重要性是否符合基本常识
```

---

### 常用调参范围参考

Random Forest：

```text
n_estimators: 100, 200, 300, 500
max_depth: None, 6, 8, 12, 16
min_samples_leaf: 1, 2, 3, 5
max_features: sqrt, log2, None
```

XGBoost：

```text
n_estimators: 100, 300, 500, 800
learning_rate: 0.01, 0.03, 0.05, 0.1
max_depth: 3, 4, 5, 6
subsample: 0.7, 0.85, 1.0
colsample_bytree: 0.7, 0.85, 1.0
reg_lambda: 0.1, 1.0, 5.0, 10.0
reg_alpha: 0.0, 0.1, 1.0
```

SVR：

```text
C: 1, 10, 50, 100
gamma: scale, auto
epsilon: 0.01, 0.05, 0.1
```

KNN：

```text
n_neighbors: 3, 5, 7, 9, 15
weights: uniform, distance
```

Gradient Boosting：

```text
n_estimators: 100, 200, 300, 500
learning_rate: 0.01, 0.03, 0.05, 0.1
max_depth: 2, 3, 4, 5
```

---

### 更规范的调参方式

当前项目中，传统机器学习参数主要写在：

```text
src/superalloy_ml/traditional_ml.py
```

如果后续需要频繁调参，建议扩展为从配置文件读取，例如在：

```text
configs/default.yaml
```

中加入：

```yaml
hyperparameters:
  random_forest_regressor:
    n_estimators: 300
    max_depth: 12
    min_samples_leaf: 2
    max_features: sqrt

  xgboost_regressor:
    n_estimators: 500
    learning_rate: 0.03
    max_depth: 5
    subsample: 0.85
    colsample_bytree: 0.85

  svr:
    C: 50.0
    epsilon: 0.02
    gamma: scale
```

这样后续可以通过不同配置文件管理不同实验：

```text
configs/random_forest_large.yaml
configs/xgboost_small_lr.yaml
configs/svr_rbf.yaml
```

运行时指定配置：

```bash
python scripts/run_traditional_ml.py --config configs/xgboost_small_lr.yaml
```

---

## 使用合成数据

默认配置使用合成数据。

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

然后运行：

```bash
python scripts/run_data_check.py --data-path data/raw/my_superalloy_data.csv --target creep_log_life_h
```

运行传统机器学习：

```bash
python scripts/run_traditional_ml.py \
  --data-path data/raw/my_superalloy_data.csv \
  --regression-target creep_log_life_h
```

单独运行某个模型：

```bash
python scripts/run_single_model.py \
  --task regression \
  --model-name random_forest_regressor \
  --data-path data/raw/my_superalloy_data.csv \
  --target creep_log_life_h
```

自动全流程使用外部 CSV：

```bash
python scripts/run_all.py \
  --data-path data/raw/my_superalloy_data.csv \
  --target creep_log_life_h \
  --skip-shap \
  --no-xgboost
```

---

## 使用外部 Excel 数据

将数据放入：

```text
data/raw/my_superalloy_data.xlsx
```

运行传统机器学习：

```bash
python scripts/run_traditional_ml.py \
  --data-path data/raw/my_superalloy_data.xlsx \
  --regression-target creep_log_life_h
```

自动全流程使用外部 Excel：

```bash
python scripts/run_all.py \
  --data-path data/raw/my_superalloy_data.xlsx \
  --target creep_log_life_h \
  --skip-shap \
  --no-xgboost
```

---

## 推荐表格字段

一行代表一个样本。

### 元素成分

```text
Ni, Co, Cr, Al, Ti, Ta, W, Mo, Re, Nb, Hf, C, B
```

### 热处理参数

```text
solution_temp_c
solution_time_h
aging_temp_c
aging_time_h
cooling_rate_c_s
```

### 测试条件

```text
test_temp_c
stress_mpa
```

### 回归目标

```text
yield_strength_mpa
tensile_strength_mpa
creep_log_life_h
oxidation_mass_gain_mg_cm2
```

### 分类目标

```text
creep_life_class
```

目标列可以不全有。只要运行时指定的目标列存在即可。

---

## 常见任务示例

### 任务 1：预测蠕变寿命

目标列：

```text
creep_log_life_h
```

运行：

```bash
python scripts/run_traditional_ml.py --regression-target creep_log_life_h
```

单独使用随机森林：

```bash
python scripts/run_single_model.py \
  --task regression \
  --model-name random_forest_regressor \
  --target creep_log_life_h
```

---

### 任务 2：预测屈服强度

目标列：

```text
yield_strength_mpa
```

运行：

```bash
python scripts/run_traditional_ml.py --regression-target yield_strength_mpa
```

单独使用 XGBoost：

```bash
python scripts/run_single_model.py \
  --task regression \
  --model-name xgboost_regressor \
  --target yield_strength_mpa
```

---

### 任务 3：预测抗拉强度

目标列：

```text
tensile_strength_mpa
```

运行：

```bash
python scripts/run_traditional_ml.py --regression-target tensile_strength_mpa
```

---

### 任务 4：预测氧化增重

目标列：

```text
oxidation_mass_gain_mg_cm2
```

运行：

```bash
python scripts/run_traditional_ml.py --regression-target oxidation_mass_gain_mg_cm2
```

---

### 任务 5：蠕变寿命等级分类

目标列：

```text
creep_life_class
```

运行：

```bash
python scripts/run_traditional_ml.py --classification-target creep_life_class
```

单独使用逻辑回归：

```bash
python scripts/run_single_model.py \
  --task classification \
  --model-name logistic_regression \
  --target creep_life_class
```

---

### 任务 6：合金样本聚类

运行：

```bash
python scripts/run_clustering.py
```

单独使用 KMeans：

```bash
python scripts/run_single_model.py \
  --task clustering \
  --model-name kmeans \
  --n-clusters 3
```

---

### 任务 7：合金样本二维可视化

运行 PCA 和 t-SNE：

```bash
python scripts/run_dimensionality_reduction.py
```

只运行 PCA：

```bash
python scripts/run_single_model.py --task reduction --model-name pca
```

只运行 t-SNE：

```bash
python scripts/run_single_model.py --task reduction --model-name tsne
```

---

### 任务 8：查看哪些特征影响预测结果

运行解释性分析：

```bash
python scripts/run_shap_analysis.py --skip-shap
```

如果安装了 SHAP：

```bash
python scripts/run_shap_analysis.py
```

输出：

```text
outputs/reports/explainability/
outputs/figures/explainability/
```

---

## 输出文件说明

运行完成后，结果保存在：

```text
outputs/
```

### 模型文件

```text
outputs/models/
```

常见文件：

```text
ridge.joblib
random_forest_regressor.joblib
xgboost_regressor.joblib
mlp_regressor.pt
```

### 结果表

```text
outputs/reports/
```

常见文件：

```text
traditional_regression_test_results.csv
traditional_regression_cv_results.csv
traditional_classification_test_results.csv
traditional_classification_cv_results.csv
clustering_summary.csv
dimensionality_reduction_summary.csv
summary_report.md
summary.json
```

### 图像

```text
outputs/figures/
```

常见文件：

```text
traditional_regression_r2.png
traditional_regression_rmse.png
traditional_regression_prediction_vs_actual.png
traditional_regression_residuals.png
traditional_classification_confusion_matrix.png
pca_embedding.png
tsne_embedding.png
shap_summary.png
shap_bar.png
```

### 预测结果

```text
outputs/predictions/
```

常见文件：

```text
traditional_regression_predictions.csv
traditional_classification_predictions.csv
clustering_assignments.csv
pca_embedding.csv
tsne_embedding.csv
```

---

## Makefile 快捷命令

如果系统支持 `make`，可以使用：

```bash
make install
make data
make check
make features
make traditional
make clustering
make reduction
make shap
make all
make test
```

没有 `make` 的环境可以直接运行 Python 命令。

---

## 测试项目

运行全部测试：

```bash
pytest tests/
```

显示详细信息：

```bash
pytest tests/ -v
```

检查项目结构：

```bash
python scripts/check_project_structure.py
```

如果没有安装 PyTorch，深度学习测试会自动跳过。

---

## 推荐检查顺序

第一次运行项目时，建议按下面顺序：

```bash
python scripts/check_project_structure.py
python scripts/make_dataset.py
python scripts/run_data_check.py
python scripts/run_feature_engineering.py
python scripts/run_traditional_ml.py --no-xgboost
python scripts/run_single_model.py --task regression --model-name random_forest_regressor
python scripts/run_clustering.py
python scripts/run_dimensionality_reduction.py
python scripts/run_shap_analysis.py --skip-shap
python scripts/run_evaluation.py
pytest tests/ -v
```

如果这些都能通过，说明基础流程正常。

---

## 注意事项

- 合成数据只用于流程验证和代码演示。
- 外部数据建议先统一单位和字段名称。
- 元素成分默认按百分比处理。
- 如果成分总和不等于 100，可以在特征工程中自动归一化。
- 小样本数据建议重点查看交叉验证结果。
- 树模型适合表格数据，SVM 和 KNN 对标准化更敏感。
- SHAP 分析主要用于解释模型，而不是证明因果关系。
- PyTorch 相关功能需要单独安装 `requirements-dl.txt`。
- SHAP 相关功能建议单独安装 `requirements-optional.txt`。

## 局限性与后续改进

当前项目默认使用合成数据，主要用于验证流程和展示方法，不能直接代表真实高温合金实验规律。真实数据使用前需要统一字段名称、单位、实验条件，并检查缺失值、异常值和重复样本。

项目目前主要面向表格数据，适合回归、分类、聚类、降维、模型解释和结果报告。对于显微组织图像分割、原始时序信号建模、文献数据抽取、主动学习、多目标优化和模型部署等任务，还需要进一步扩展。

当前特征工程是高温合金简化版本，SHAP 和特征重要性分析只能解释模型相关性，不能直接证明材料因果关系。后续可继续加入真实数据示例、通用表格模式、自动超参数优化、图像/时序建模、多目标优化、主动学习和批量预测部署功能。