# B6pro 状态

## 结论（当前）

- B7 保底：本地 **0.702704955** / 公开 **0.707**
- **新 closest**：**0.70544811**（mean(aging,gap,keepx) + region blend wo=0.15）
- 相对 B7 **+0.00274**；距 0.71 缺口 ≈ **0.00455**
- 产物：`artifacts/b6pro_long_best/`、`submissions/b6pro_closest/`

## 本轮进行中

- LGBM resid：nested 0.70497（未超 closest）；long corr(max3)≈0.52 但 slice≈0.60 过弱
- CatBoost resid_cb / KNN / region_local / resid_corr：排队训练中
- plus_gap2：新 plus≈0.68 < ref 0.688，预期难抬 max3

## 业务缺口（主杠杆）

- long AUC 0.668→≈0.68 即可整体过 0.71（pair-swap 灵敏度）
- 同 region LL pair acc≈0.63；错对呈 anti-monotonic（低 days/高 condition 却索赔）
- 优先异构残差 / 区域专模 / KNN 局部索赔率，避免再堆 corr≈0.99 CatBoost
