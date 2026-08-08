# B6pro 状态（更正）

## 公开榜反馈（必须正视）

- 过拟合版本本地宣称 **0.710071**，提交后公开榜 **0.70208**
- **低于 B7 公开 0.707** → 该版本 **作废，禁止再交**
- 根因嫌疑：单 seed 樱桃采摘（仅 2027 过线）、多层同标签 OOF 堆叠、ultra 小样本嵌套 α

## 当前可提交（保底）

- **立刻交 B7**：`submissions/b6pro_closest/submission_SUBMIT_THIS.csv`
- 同源：`reference/b7_closest/submission_b7_closest_honest.csv`
- 本地 nested **0.702705** / 公开已验证 **0.707**

## 诚实 closest（重新定义）

在修复协议落地前，**不以任何 >B7 的本地虚高分为可交付 closest**。  
可交付条件（同时满足）：

1. 外层嵌套 OOF ≥ 目标，且 **≥4 seeds 等权**（禁止单 seed 过线）
2. **重复外层 seed（≥3 组 SKF random_state）** 均值过线，单次波动计入报告
3. 相对 B7：overall 提升的同时，**不允许**仅靠 n&lt;2k 切片高 α patch 抬整体
4. 与 B7 test 分数相关不能异常漂移到“只改小众行却大幅改分布”
5. 未达以上 → fallback **B7**

## 下一目标

在不过拟合约束下把 **可泛化** 本地 nested 真正推向 0.71；未达成不停。  
独立监察 agent 审核每次 promote。
