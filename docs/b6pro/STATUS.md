# B6pro 状态

## 结论（当前）

- 你的 B7 提交 `submission_b7_closest_honest.csv` **就是** `max(gap,gap_bag,plus_v10)`  
- 本地 **0.702704955** → 公开 **0.707**（已核对 OOF 与冻结文件一致）  
- 冲本地 **0.71**：目前 closest 仍为 **0.70270496**，缺口 ≈0.0073  
- 已试门控/专模/sink/加权/FLAML/TabPFN(需授权)/MLP/stack 等，**均未超过 max3**

## 继续方向

寻找 **solo≳0.695 且 corr(max3)≲0.90** 的真正异构臂；同质 CatBoost 变体已耗尽。
