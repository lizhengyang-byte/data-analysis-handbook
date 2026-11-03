# 配套代码目录

本目录包含《数据分析实战手册》各章节的可运行代码示例。

## 环境要求

```bash
pip install pandas numpy scikit-learn matplotlib seaborn statsmodels openpyxl
```

推荐 Python 3.9+。

## 目录说明

| 目录 | 对应章节 | 内容 |
|------|---------|------|
| `07-sql/` | 第07章 | SQL 常用查询模板 |
| `08-python-pandas/` | 第08章 | NumPy/Pandas 常用操作演示 |
| `16-projects/` | 第16章 | 6个完整项目实战 |
| `21-regression/` | 第21章 | 回归算法实战（房价预测） |
| `22-classification/` | 第22章 | 分类算法实战（客户流失预测） |
| `24-time-series/` | 第24章 | 时间序列预测（销售预测） |
| `25-anomaly-detection/` | 第25章 | 异常检测（欺诈识别） |
| `26-nlp/` | 第26章 | NLP 情感分析 |
| `27-recommender/` | 第27章 | 推荐系统（电影推荐） |

## 数据集说明

代码中使用的数据来源：

- **scikit-learn 内置数据集**：`fetch_california_housing`、`load_breast_cancer`、`make_blobs` 等，运行时自动下载
- **合成数据**：部分脚本使用 `numpy.random` 生成模拟数据，无需额外下载
- **公开数据集**：项目案例中涉及的数据集会在各自目录的 README 中说明获取方式（Kaggle 经典数据集）

所有脚本设计为可以直接 `python xxx.py` 运行，不依赖外部私有数据。
