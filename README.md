# 数据分析实战手册

> 一本从零基础到高阶实战的完整指南，涵盖数据分析全链路方法论、工具栈、业务场景与面试体系。

![Data Analysis Handbook](https://img.shields.io/badge/Data%20Analysis-Handbook-blue)
![Chapters](https://img.shields.io/badge/Chapters-18-green)
![Code Examples](https://img.shields.io/badge/Code%20Examples-200%2B-orange)
![Last Updated](https://img.shields.io/badge/Last%20Updated-2026.11-red)

---

## 关于本书

这是一本**面向实战**的数据分析手册，不是教科书，也不是 API 文档。它回答的核心问题是：

> **当你拿到一份业务数据时，从第一步到最后一份报告，每一步具体该怎么做？**

全书围绕一个完整的数据分析工作流展开：

```
明确问题 → 数据获取 → 数据清洗 → 探索分析 → 统计验证 → 可视化呈现 → 业务建议 → 自动化交付
```

每一章都包含：
- **方法论**：为什么这么做
- **操作步骤**：具体怎么做（分步到可复现）
- **代码示例**：Python / SQL 可直接运行
- **常见坑**：真实项目中踩过的坑
- **检查清单**：交付前自检

## 全书目录

| 章节 | 主题 | 核心内容 |
|------|------|----------|
| [01](01-data-analysis-foundations.md) | 数据分析基础与思维框架 | 数据分析是什么、分析思维、问题拆解、经典模型 |
| [02](02-data-collection.md) | 数据获取与采集 | SQL取数、爬虫、API、日志、埋点、第三方数据 |
| [03](03-data-cleaning.md) | 数据清洗与预处理 | 缺失值、异常值、重复值、格式统一、特征工程基础 |
| [04](04-exploratory-data-analysis.md) | 探索性数据分析（EDA） | 单变量/双变量/多变量分析、分布、相关、分组 |
| [05](05-statistical-analysis.md) | 统计分析与假设检验 | 描述统计、推断统计、t检验、卡方、方差分析、非参数 |
| [06](06-data-visualization.md) | 数据可视化 | 图表选型、Matplotlib/Seaborn/Plotly、配色、仪表板 |
| [07](07-sql-for-data-analysis.md) | SQL数据分析实战 | 窗口函数、CTE、性能优化、常见查询模板 |
| [08](08-python-pandas-numpy.md) | Python数据分析（Pandas/NumPy） | DataFrame操作、分组聚合、合并、时间序列、性能优化 |
| [09](09-business-analysis-methodology.md) | 业务数据分析方法论 | 指标体系、漏斗分析、留存分析、RFM、杜邦分析 |
| [10](10-ab-testing-causal-inference.md) | A/B测试与因果推断 | 实验设计、样本量、显著性、辛普森悖论、DID、PSM |
| [11](11-user-behavior-analysis.md) | 用户行为分析 | 用户画像、路径分析、Session、活跃分析、用户分层 |
| [12](12-sales-revenue-analysis.md) | 销售与营收分析 | GMV拆解、客单价、复购、渠道分析、同比环比 |
| [13](13-operations-analysis.md) | 运营数据分析 | 活动分析、Push/推送效果、内容分析、增长黑客指标 |
| [14](14-reporting-automation.md) | 数据报表与自动化 | 报表体系、定时任务、邮件推送、Tableau/Power BI |
| [15](15-ml-for-data-analysis.md) | 机器学习在数据分析中的应用 | 分类回归聚类、异常检测、预测、特征重要性 |
| [16](16-real-world-case-studies.md) | 真实项目实战案例 | 电商、内容、金融、SaaS 四个完整案例全流程 |
| [17](17-tools-ecosystem.md) | 数据分析工具生态 | Excel→SQL→Python→BI→数仓全栈工具链对比与选型 |
| [18](18-career-interview-guide.md) | 职业发展与面试指南 | 技能树、简历、面试题、晋升路径、薪资参考 |

## 适合谁读

- **转行/应届生**：从第一章顺序读到第十八章，配合代码动手练
- **业务分析师/运营**：重点读 09-13 章业务方法论
- **数据分析师**：查漏补缺，重点读 05、10、15 章高阶内容
- **数据产品/PM**：重点读 09、14、18 章

## 环境准备

```bash
# Python 环境推荐
python >= 3.10
pandas >= 2.0
numpy >= 1.24
matplotlib >= 3.7
seaborn >= 0.12
scipy >= 1.10
scikit-learn >= 1.3
plotly >= 5.15
sqlalchemy >= 2.0

# 一键安装
pip install pandas numpy matplotlib seaborn scipy scikit-learn plotly sqlalchemy jupyter
```

## 开源协议

MIT License — 可自由使用、修改、分发，保留原作者署名即可。

---

*如果这本手册对你有帮助，欢迎 Star ⭐ 和贡献 PR。*
