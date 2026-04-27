from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from superalloy_ml.evaluation import classification_metrics, regression_metrics
from superalloy_ml.preprocessing import create_preprocessor
from superalloy_ml.utils import ensure_dir

TaskType = Literal["regression", "classification"]


def try_import_xgboost():
    """
    延迟导入 XGBoost。

    如果环境中没有安装 XGBoost，可以在调用模型集合时关闭 include_xgboost。
    """
    try:
        from xgboost import XGBClassifier, XGBRegressor

        return XGBRegressor, XGBClassifier
    except Exception:
        return None, None


def get_regression_models(
    random_state: int = 42,
    include_xgboost: bool = True,
) -> dict[str, object]:
    """
    获取回归模型集合。
    """
    models: dict[str, object] = {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=1.0, random_state=random_state),
        "lasso": Lasso(alpha=0.001, max_iter=5000, random_state=random_state),
        "elasticnet": ElasticNet(
            alpha=0.001,
            l1_ratio=0.5,
            max_iter=5000,
            random_state=random_state,
        ),
        "knn_regressor": KNeighborsRegressor(n_neighbors=7),
        "svr": SVR(kernel="rbf", C=10.0, epsilon=0.05),
        "decision_tree_regressor": DecisionTreeRegressor(
            max_depth=8,
            min_samples_leaf=3,
            random_state=random_state,
        ),
        "random_forest_regressor": RandomForestRegressor(
            n_estimators=160,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state,
        ),
        "extra_trees_regressor": ExtraTreesRegressor(
            n_estimators=160,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state,
        ),
        "gradient_boosting_regressor": GradientBoostingRegressor(
            n_estimators=180,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
        ),
    }

    if include_xgboost:
        XGBRegressor, _ = try_import_xgboost()

        if XGBRegressor is not None:
            models["xgboost_regressor"] = XGBRegressor(
                n_estimators=180,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                random_state=random_state,
                n_jobs=1,
                verbosity=0,
            )

    return models


def get_classification_models(
    random_state: int = 42,
    include_xgboost: bool = True,
) -> dict[str, object]:
    """
    获取分类模型集合。
    """
    models: dict[str, object] = {
        "logistic_regression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "knn_classifier": KNeighborsClassifier(n_neighbors=7),
        "svm_classifier": SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=random_state,
        ),
        "decision_tree_classifier": DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest_classifier": RandomForestClassifier(
            n_estimators=160,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),
        "extra_trees_classifier": ExtraTreesClassifier(
            n_estimators=160,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),
        "gradient_boosting_classifier": GradientBoostingClassifier(
            n_estimators=180,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
        ),
    }

    if include_xgboost:
        _, XGBClassifier = try_import_xgboost()

        if XGBClassifier is not None:
            models["xgboost_classifier"] = XGBClassifier(
                n_estimators=180,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=random_state,
                n_jobs=1,
                verbosity=0,
            )

    return models


def get_models(
    task: TaskType,
    random_state: int = 42,
    include_xgboost: bool = True,
) -> dict[str, object]:
    """
    根据任务类型获取模型集合。
    """
    if task == "regression":
        return get_regression_models(
            random_state=random_state,
            include_xgboost=include_xgboost,
        )

    if task == "classification":
        return get_classification_models(
            random_state=random_state,
            include_xgboost=include_xgboost,
        )

    raise ValueError(f"未知任务类型: {task}")


def build_pipeline(
    X: pd.DataFrame,
    model: object,
    scaler: str = "standard",
    numeric_strategy: str = "median",
    categorical_strategy: str = "most_frequent",
) -> Pipeline:
    """
    构建 sklearn 流水线。

    统一结构：
    预处理 -> 模型
    """
    preprocessor = create_preprocessor(
        X,
        scaler=scaler,
        numeric_strategy=numeric_strategy,
        categorical_strategy=categorical_strategy,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def sort_model_results(results: pd.DataFrame, task: TaskType) -> pd.DataFrame:
    """
    对模型结果排序。
    """
    if results.empty:
        return results

    if task == "regression":
        sort_columns = [column for column in ["r2", "rmse"] if column in results.columns]
        ascending = [False, True][: len(sort_columns)]
    else:
        sort_columns = [column for column in ["f1", "accuracy"] if column in results.columns]
        ascending = [False, False][: len(sort_columns)]

    if not sort_columns:
        return results.reset_index(drop=True)

    return results.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)


def get_best_model_name(results: pd.DataFrame, task: TaskType) -> str:
    """
    获取结果表中的最佳模型名称。
    """
    sorted_results = sort_model_results(results, task=task)

    if sorted_results.empty:
        raise ValueError("结果表为空，无法选择模型。")

    return str(sorted_results.iloc[0]["model_name"])


def train_test_evaluate_models(
    X: pd.DataFrame,
    y: pd.Series,
    task: TaskType,
    test_size: float = 0.2,
    random_state: int = 42,
    include_xgboost: bool = True,
    scaler: str = "standard",
    numeric_strategy: str = "median",
    categorical_strategy: str = "most_frequent",
    save_model_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    """
    训练并评估一组模型。

    返回：
    - 结果表
    - 已训练模型字典
    """
    stratify = y if task == "classification" else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    models = get_models(
        task=task,
        random_state=random_state,
        include_xgboost=include_xgboost,
    )

    rows = []
    trained_models: dict[str, Pipeline] = {}

    if save_model_dir is not None:
        save_model_dir = ensure_dir(save_model_dir)

    for model_name, model in models.items():
        start_time = time.time()

        pipeline = build_pipeline(
            X_train,
            model=model,
            scaler=scaler,
            numeric_strategy=numeric_strategy,
            categorical_strategy=categorical_strategy,
        )
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        elapsed = time.time() - start_time

        if task == "regression":
            metrics = regression_metrics(y_test, y_pred)
        else:
            metrics = classification_metrics(y_test, y_pred)

        row = {
            "model_name": model_name,
            "task": task,
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "training_time_sec": float(elapsed),
        }
        row.update(metrics)
        rows.append(row)

        trained_models[model_name] = pipeline

        if save_model_dir is not None:
            save_model(pipeline, Path(save_model_dir) / f"{model_name}.joblib")

    results = pd.DataFrame(rows)
    results = sort_model_results(results, task=task)

    return results, trained_models


def cross_validate_models(
    X: pd.DataFrame,
    y: pd.Series,
    task: TaskType,
    cv: int = 5,
    random_state: int = 42,
    include_xgboost: bool = True,
    scaler: str = "standard",
    scoring: str | None = None,
    numeric_strategy: str = "median",
    categorical_strategy: str = "most_frequent",
) -> pd.DataFrame:
    """
    对多个模型进行交叉验证。
    """
    if scoring is None:
        scoring = "neg_root_mean_squared_error" if task == "regression" else "accuracy"

    if task == "classification":
        splitter = StratifiedKFold(
            n_splits=cv,
            shuffle=True,
            random_state=random_state,
        )
    else:
        splitter = KFold(
            n_splits=cv,
            shuffle=True,
            random_state=random_state,
        )

    models = get_models(
        task=task,
        random_state=random_state,
        include_xgboost=include_xgboost,
    )

    rows = []

    for model_name, model in models.items():
        pipeline = build_pipeline(
            X,
            model=model,
            scaler=scaler,
            numeric_strategy=numeric_strategy,
            categorical_strategy=categorical_strategy,
        )

        scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=splitter,
            scoring=scoring,
            n_jobs=1,
        )

        row = {
            "model_name": model_name,
            "task": task,
            "scoring": scoring,
            "cv_mean": float(scores.mean()),
            "cv_std": float(scores.std()),
        }

        if scoring.startswith("neg_"):
            row["cv_mean_positive"] = float(-scores.mean())
            row["cv_std_positive"] = float(scores.std())

        rows.append(row)

    results = pd.DataFrame(rows)

    if scoring.startswith("neg_"):
        results = results.sort_values("cv_mean_positive", ascending=True).reset_index(drop=True)
    else:
        results = results.sort_values("cv_mean", ascending=False).reset_index(drop=True)

    return results


def save_model(model: object, path: str | Path) -> None:
    """
    保存模型。
    """
    path = Path(path)
    ensure_dir(path.parent)
    joblib.dump(model, path)


def load_model(path: str | Path) -> object:
    """
    加载模型。
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"模型文件不存在: {path}")

    return joblib.load(path)