# B6 计划（基于冻结的 B5，不改动 B5 交付）

## 基线（冻结）
- 代号：`b5_focus_8seed_newdata`
- 分支：`cursor/claim-auc698-council-a5f5`
- 诚实 pooled OOF：**0.69817454**
- 提交：`submissions/b5_frozen/submission_b5_8seed.csv`

## B6 目标
- 诚实本地 pooled OOF **≥ 0.70**
- 协议：折内 FE、无 TE（或仅严格 nested）、无 OOF 搜权、等权多种子、shuffled∈[0.47,0.53]
- 不修改 B5 产物与代码路径语义；B6 以增量模块/脚本交付

## 抬分方向（预注册）
1. B5 主臂保留 + 异构臂（Lossguide / 解析增强 / 物理残差）等权或等权 rank（二选一预注册）
2. 种子扩展（≥8，目标 12）等权 bagging
3. 早停乐观对照：固定 iteration 无 od 的稳健臂
4. 禁止：全局 TE、OOF 网格搜权、旧预测包、测集伪标签
