# 比赛说明

## 任务

车险相关表格数据上的**二分类**：预测是否索赔（`label` 0/1）。  
评价指标为 **ROC-AUC**。提交文件两列：`id,label`（`label` 为预测概率）。

## 数据规模

| 集合 | 路径 | 规模 |
|---|---|---|
| 训练 | `data/train.csv` | 14930 × 45（含 label） |
| 测试 | `data/test.csv` | 6398 × 44 |
| 提交模板 | `data/submit_sample.csv` | 6398 × 2 |

特征含：地域/车源/版本/天数/车况、条款开关、以及若干数值与字符串字段等（详见 CSV 表头）。

## 本仓库对应方案

**B7 closest**：`max(B6_gap, B6_gap_bag, V10_plus)`  
- 本地诚实 AUC：**0.702704955**  
- 公开榜：**0.70722**  
- 提交：`submissions/submission_b7_closest_honest.csv`
