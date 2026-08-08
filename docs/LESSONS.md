# B7 实战经验与踩坑录

> 目的：把 B5→B6→V10→B7 迭代里**真正有用 / 浪费时间 / 有坑**的结论沉淀下来，避免后人重复交学费。  
> 权威分数：本地 closest nested **0.702705**；公开榜（`submission_b7_closest_honest.csv`）**0.70722**。

---

## 1. 什么真正有效（值得做）

| 动作 | 为什么有效 | 量级（本地） |
|---|---|---|
| **丢掉近 ID 噪声 `x0–x18`（B5/B6 主臂）** | 主排序臂更干净；plus 再单独吃 latent | B5≈0.698 |
| **gap 猫交叉**（ratio×region/source、t3_sfx×code×days、w_pair×days…） | B5 未吃满的业务切割 | B6≈0.699 |
| **异构 plus 臂（保留 x0–x18 + root 交叉）** | 与 B5/B6 corr≈0.91–0.92，互补 | plus solo≈0.6886 |
| **预注册离散融合 + 嵌套选规则** | 可审计；避免连续搜权幻觉 | — |
| **`max` / 三臂 `max(gap,gap_bag,plus)`** | 捕捉「某臂在该样本上更敢赌」的排序 | 本地 **0.7027**；公开 **0.70722** |
| **plus 必须 10fold×多 seed** | 5fold×1seed 会严重低估 plus（~0.68） | 差约 0.008–0.01 |
| **shuffled_plus→max 崩盘检查** | 证明 max 增益来自真实排序而非空壳公式 | ~0.64 |

**公开榜解读：** 本地 0.7027 → 公开 **0.70722**（约 +0.0045）。与 V10 当时「本地 nested 低于公开」同向，支持异构 max 在公开集仍有效，但**不能**把公开分说成 CV。

---

## 2. 有坑（看起来聪明，实际伤人）

### 2.1 融合 / 协议坑

- **凸组合门控**（学「何时信 plus」再 soft blend）：任意 `g·plus+(1-g)·B6` **点式不超过** `max(B6,plus)`，再 `max(soft,s1)` 等于原地踏步。  
- **事后扩融合规则却不披露**：max3 若从 disclosure 悄悄升格主报，审计会给 **CONDITIONAL**。开跑前写进预注册集合。  
- **连续 OOF 搜权 / logistic stack**：本地常假高或反而掉（B7 stack≈0.6969），且过不了诚实协议。  
- **用 full-data 挑最高规则冒充 nested**：禁止；主报必须是嵌套折选定规则后的 OOF。  
- **校准（isotonic/Platt）再 max**：AUC 是秩指标；乱校准常伤害 max 融合。

### 2.2 特征 / CV 坑

- **全局 TE / 全量 fit 再 CV**：分数虚高，一查就挂。  
- **OOF 叠层泄漏**：把「别套 CV 的全量 OOF」当特征再训残差，折内 AUC 好看、一融合就掉（B7 resid nested≈0.697）。要 nested 重算 stage1。  
- **稀疏三阶交叉 / 高 gap TE**（如过碎的 `region_car_d5`）：诊断相关高，上模型易过拟合或无增益。  
- **plus 只跑 5fold×1seed 就下结论**：会误判「plus 不行」；V10 强度来自 10×4 池化。

### 2.3 训练坑

- **`auto_class_weights=Balanced` / 强 FN 加权**：本任务正例≈10%，强平衡常**毁主排序**（gap_bal max 反降）。  
- **Early stopping on valid**：OOF 有轻度乐观偏置——要披露，不要当「无偏 CV」。  
- **同质 CatBoost 微扰当新臂**（仅改 bagging/RSM/seed）：corr 常 >0.98，`max` 几乎不抬。  
- **把公开榜当调参反馈环**：过拟合榜单，本地诚实协议也会 REJECT。

---

## 3. 浪费时间（投入大、回报接近 0）

| 动作 | 结果摘要 | 建议 |
|---|---|---|
| 残差 CatBoost corrector（stage1 作特征） | nested 0.697 < fuse0 | 别做非 nested 叠层 |
| soft gate / disagree_max / softmax 融合 | ≤ fuse0 | 两臂场景优先 max/mean 族 |
| plus_mine（硬塞 gap/FN 交叉进 plus） | solo 0.686，弱于 V10 plus | plus 保持异构，勿强行同化 |
| plus H3 / bag 集成再 max B6 | ~0.7017 < closest | 边际不够 |
| hybrid（gap FE + 强行加回 x0–x18） | corr(B6)≈0.987 | 破坏异构 |
| LGB / XGB / EBM / 浅 MLP 当第三臂 | solo 多在 0.64–0.67 | 不够近强度，拖融合 |
| Balanced / midband 样本加权专家 | 主 AUC 掉 | 除非单独做召回臂且验证 max 有增益 |
| 嵌套残差 TE 校正 | ~0.686 | 无增益 |
| 12 seed 狂扩同质主臂 | 相对 8 seed 抬分极小 | 优先找异构信号 |
| 复杂 meta-stack | 常低于简单 max | 先穷尽离散融合 |

---

## 4. 哪些模型 / 配方效果不好（本数据）

相对「B6 CatBoost + V10 plus CatBoost + max」：

- **LightGBM on gap FE**：臂弱（~0.67），融合无超 closest  
- **XGBoost 频编矩阵**：更弱（~0.65）  
- **EBM (interpret)**：~0.64 量级  
- **sklearn MLP**：~0.57，基本无效  
- **RealMLP / TabKit**：环境未作为主路径；不要默认赌 NN 翻盘  
- **Lossguide / 过深树 / 过强正则乱扫**：B6 期多为噪声  
- **业务 lean 臂单独硬刚**：常弱于 focus+gap  
- **V9 级方案**：本地显著弱于 B5/B6/V10，勿回退

---

## 5. 推荐工作流（省时间版）

1. **先定协议**：数据 SHA、折内 FE、禁止全局 TE、预注册融合规则、主报 nested。  
2. **做强主臂**（CatBoost + 业务交叉），8 seed 够用。  
3. **刻意做异构臂**（不同特征世界观，corr 目标约 0.90–0.93），plus 类必须多 fold×多种子。  
4. **只在预注册集合里嵌套选融合**；两臂/三臂优先试 `max` / `mean` / `power*`。  
5. **每次声称增益**：复算 OOF + shuffled 崩盘 + 与冻结基线对比。  
6. **公开榜只验收、不闭环调参**；本地与公开差 0.003–0.005 都可能出现。  
7. 冲更高分时：找 **solo≳0.69 且 corr≲0.90** 的新信号；不要再堆同质 CatBoost。

---

## 6. 数字速查

| 项 | 值 |
|---|---:|
| B5 frozen pooled | 0.69817454 |
| B6 equal (gap,gap_bag) | 0.69897470 |
| V10 nested max(B5,plus) | 0.70131497 |
| V10 公开榜（历史） | 0.70570 |
| B7 closest 本地 | **0.70270496** |
| B7 closest **公开榜** | **0.70722** |
| 本地→公开（B7） | ≈ **+0.0045** |

---

## 7. 一句话总结

**异构 > 同质堆料；max 有用但要诚实嵌套；协议红线比多挖 0.001 分重要；公开榜 0.70722 证明 B7 closest 可上榜，但调参仍必须以本地诚实口径为准。**
