import pandas as pd

from superalloy_ml.reporting import (
    create_output_dirs,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_prediction_vs_actual,
    plot_residuals,
    save_table,
    write_markdown_report,
)


def test_create_output_dirs(tmp_path):
    dirs = create_output_dirs(tmp_path)

    assert dirs["base"].exists()
    assert dirs["models"].exists()
    assert dirs["reports"].exists()
    assert dirs["figures"].exists()
    assert dirs["predictions"].exists()


def test_save_table(tmp_path):
    df = pd.DataFrame(
        {
            "model_name": ["a", "b"],
            "r2": [0.8, 0.9],
        }
    )

    path = tmp_path / "reports" / "result.csv"
    save_table(df, path)

    assert path.exists()


def test_plot_model_comparison(tmp_path):
    df = pd.DataFrame(
        {
            "model_name": ["a", "b", "c"],
            "r2": [0.7, 0.8, 0.9],
        }
    )

    path = tmp_path / "figures" / "comparison.png"

    plot_model_comparison(
        results=df,
        metric="r2",
        path=path,
        title="模型对比",
        higher_is_better=True,
    )

    assert path.exists()


def test_plot_prediction_vs_actual(tmp_path):
    path = tmp_path / "figures" / "prediction_vs_actual.png"

    plot_prediction_vs_actual(
        y_true=[1.0, 2.0, 3.0],
        y_pred=[1.1, 1.9, 3.2],
        path=path,
    )

    assert path.exists()


def test_plot_residuals(tmp_path):
    path = tmp_path / "figures" / "residuals.png"

    plot_residuals(
        y_true=[1.0, 2.0, 3.0],
        y_pred=[1.1, 1.9, 3.2],
        path=path,
    )

    assert path.exists()


def test_plot_confusion_matrix(tmp_path):
    path = tmp_path / "figures" / "confusion_matrix.png"

    plot_confusion_matrix(
        y_true=[0, 1, 1, 2],
        y_pred=[0, 1, 0, 2],
        path=path,
    )

    assert path.exists()


def test_write_markdown_report(tmp_path):
    path = tmp_path / "reports" / "report.md"

    write_markdown_report(
        path=path,
        title="测试报告",
        sections={
            "说明": "这是一个测试报告。",
            "结果": "文件生成成功。",
        },
    )

    assert path.exists()