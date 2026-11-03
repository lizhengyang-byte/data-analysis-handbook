# 示例数据说明

本目录存放手册中代码示例涉及的数据说明。

## 数据来源

| 数据集 | 来源 | 说明 |
|--------|------|------|
| California Housing | scikit-learn 内置 `fetch_california_housing` | 加州房价，回归任务 |
| Breast Cancer Wisconsin | scikit-learn 内置 `load_breast_cancer` | 乳腺癌诊断，二分类演示 |
| 合成销售数据 | 代码内 `numpy.random` 生成 | 时间序列演示 |
| 合成交易数据 | 代码内生成 | 异常检测演示 |
| 合成评论数据 | 代码内列表定义 | NLP情感分析演示 |

## 不包含私有数据

为保证仓库公开可用，所有示例数据均为：
1. scikit-learn 内置公开数据集
2. 代码内用 `numpy.random` 生成的模拟数据
3. 少量手写的演示文本

如果需要使用真实业务数据，请自行替换代码中的数据加载部分，保持接口（X, y 的 DataFrame/ndarray 格式）一致即可。
