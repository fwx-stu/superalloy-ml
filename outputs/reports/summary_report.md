# 模型结果汇总报告

## 整体说明

本报告汇总项目运行过程中生成的主要结果。如果某一部分显示暂无结果文件，说明对应脚本尚未运行，或该流程被跳过。

## 最佳回归模型

- model_name: linear_regression
- task: regression
- train_size: 960
- test_size: 240
- training_time_sec: 0.0310559272766113
- mae: 0.1925167655132396
- mse: 0.0612899628452123
- rmse: 0.2475680973898138
- r2: 0.9329301592151034
- mape: 0.027883074119611
- max_error: 0.7530448043181348

## 最佳分类模型

- model_name: logistic_regression
- task: classification
- train_size: 960
- test_size: 240
- training_time_sec: 0.0413331985473632
- accuracy: 0.8583333333333333
- precision: 0.8580837707111248
- recall: 0.8580837707111248
- f1: 0.8580837707111248

## 传统机器学习回归结果

| model_name                  | task       |   train_size |   test_size |   training_time_sec |      mae |       mse |     rmse |       r2 |      mape |   max_error |
|:----------------------------|:-----------|-------------:|------------:|--------------------:|---------:|----------:|---------:|---------:|----------:|------------:|
| linear_regression           | regression |          960 |         240 |          0.0310559  | 0.192517 | 0.06129   | 0.247568 | 0.93293  | 0.0278831 |    0.753045 |
| ridge                       | regression |          960 |         240 |          0.00682616 | 0.19335  | 0.0614841 | 0.24796  | 0.932718 | 0.0280221 |    0.754867 |
| elasticnet                  | regression |          960 |         240 |          0.0331898  | 0.194069 | 0.0617986 | 0.248593 | 0.932374 | 0.0281381 |    0.766775 |
| lasso                       | regression |          960 |         240 |          0.0163424  | 0.19391  | 0.0618308 | 0.248658 | 0.932338 | 0.0281237 |    0.764427 |
| gradient_boosting_regressor | regression |          960 |         240 |          2.63336    | 0.216986 | 0.080416  | 0.283577 | 0.912    | 0.0316848 |    0.894416 |
| extra_trees_regressor       | regression |          960 |         240 |          0.159706   | 0.237185 | 0.0933089 | 0.305465 | 0.897892 | 0.0344617 |    0.936942 |
| random_forest_regressor     | regression |          960 |         240 |          0.372197   | 0.247587 | 0.099735  | 0.315809 | 0.89086  | 0.0360206 |    0.91952  |
| svr                         | regression |          960 |         240 |          0.0752134  | 0.284989 | 0.140957  | 0.375442 | 0.845751 | 0.041583  |    1.72287  |
| decision_tree_regressor     | regression |          960 |         240 |          0.0286484  | 0.398061 | 0.242173  | 0.49211  | 0.73499  | 0.0575723 |    1.41719  |
| knn_regressor               | regression |          960 |         240 |          2.86469    | 0.410022 | 0.281961  | 0.531    | 0.691449 | 0.0586622 |    2.11665  |

## 传统机器学习回归交叉验证

| model_name                  | task       | scoring                     |   cv_mean |     cv_std |   cv_mean_positive |   cv_std_positive |
|:----------------------------|:-----------|:----------------------------|----------:|-----------:|-------------------:|------------------:|
| lasso                       | regression | neg_root_mean_squared_error | -0.251702 | 0.005059   |           0.251702 |        0.005059   |
| elasticnet                  | regression | neg_root_mean_squared_error | -0.252108 | 0.00484537 |           0.252108 |        0.00484537 |
| ridge                       | regression | neg_root_mean_squared_error | -0.252878 | 0.00458103 |           0.252878 |        0.00458103 |
| linear_regression           | regression | neg_root_mean_squared_error | -0.253501 | 0.00526427 |           0.253501 |        0.00526427 |
| gradient_boosting_regressor | regression | neg_root_mean_squared_error | -0.290341 | 0.0133881  |           0.290341 |        0.0133881  |
| extra_trees_regressor       | regression | neg_root_mean_squared_error | -0.309862 | 0.0118201  |           0.309862 |        0.0118201  |
| random_forest_regressor     | regression | neg_root_mean_squared_error | -0.328276 | 0.0184932  |           0.328276 |        0.0184932  |
| svr                         | regression | neg_root_mean_squared_error | -0.368846 | 0.01775    |           0.368846 |        0.01775    |
| decision_tree_regressor     | regression | neg_root_mean_squared_error | -0.482109 | 0.0252448  |           0.482109 |        0.0252448  |
| knn_regressor               | regression | neg_root_mean_squared_error | -0.533195 | 0.0124981  |           0.533195 |        0.0124981  |

## 传统机器学习分类结果

| model_name                   | task           |   train_size |   test_size |   training_time_sec |   accuracy |   precision |   recall |       f1 |
|:-----------------------------|:---------------|-------------:|------------:|--------------------:|-----------:|------------:|---------:|---------:|
| logistic_regression          | classification |          960 |         240 |           0.0413332 |   0.858333 |    0.858084 | 0.858084 | 0.858084 |
| extra_trees_classifier       | classification |          960 |         240 |           0.360125  |   0.833333 |    0.838424 | 0.83323  | 0.835155 |
| gradient_boosting_classifier | classification |          960 |         240 |           9.17164   |   0.829167 |    0.838172 | 0.829011 | 0.831761 |
| svm_classifier               | classification |          960 |         240 |           0.308294  |   0.816667 |    0.820363 | 0.816353 | 0.817883 |
| random_forest_classifier     | classification |          960 |         240 |           0.511007  |   0.808333 |    0.815264 | 0.808223 | 0.810765 |
| decision_tree_classifier     | classification |          960 |         240 |           0.0570302 |   0.766667 |    0.762878 | 0.766029 | 0.76372  |
| knn_classifier               | classification |          960 |         240 |           0.0346024 |   0.7125   |    0.70885  | 0.711948 | 0.708037 |

## 传统机器学习分类交叉验证

| model_name                   | task           | scoring   |   cv_mean |    cv_std |
|:-----------------------------|:---------------|:----------|----------:|----------:|
| logistic_regression          | classification | accuracy  |  0.846667 | 0.0209828 |
| extra_trees_classifier       | classification | accuracy  |  0.816667 | 0.0278887 |
| gradient_boosting_classifier | classification | accuracy  |  0.815833 | 0.0423609 |
| random_forest_classifier     | classification | accuracy  |  0.7975   | 0.0249444 |
| svm_classifier               | classification | accuracy  |  0.789167 | 0.02085   |
| decision_tree_classifier     | classification | accuracy  |  0.743333 | 0.0265623 |
| knn_classifier               | classification | accuracy  |  0.6525   | 0.0350198 |

## 聚类结果

| algorithm     |   n_clusters |   n_noise |   silhouette |   calinski_harabasz |   davies_bouldin |
|:--------------|-------------:|----------:|-------------:|--------------------:|-----------------:|
| kmeans        |            3 |         0 |    0.0792788 |            105.933  |          2.81672 |
| dbscan        |            0 |      1200 |  nan         |            nan      |        nan       |
| agglomerative |            3 |         0 |    0.0262488 |             64.6187 |          3.55829 |

## 降维结果

| algorithm   |   n_components | explained_variance_ratio                   |   explained_variance_ratio_sum |   perplexity |   kl_divergence |
|:------------|---------------:|:-------------------------------------------|-------------------------------:|-------------:|----------------:|
| pca         |              2 | [0.14922279674357036, 0.11588143849886552] |                       0.265104 |          nan |        nan      |
| tsne        |              2 | nan                                        |                     nan        |           30 |          2.1257 |

## 深度学习结果

暂无结果文件。

## 输出目录

- 模型文件: `D:\机器学习算法\superalloy-ml\outputs\models`
- 结果表: `D:\机器学习算法\superalloy-ml\outputs\reports`
- 图像文件: `D:\机器学习算法\superalloy-ml\outputs\figures`
- 预测结果: `D:\机器学习算法\superalloy-ml\outputs\predictions`
