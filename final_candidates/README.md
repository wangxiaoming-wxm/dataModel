# 最终提交候选

用户已明确要求准备提交，因此这里提供两个文件；这不代表已获得 0.72 的可信证据。

## 推荐顺序

1. `submission_1_safe_v1_anchor.csv`
   - 本地 OOF：`0.69295958`
   - 已知同预测公开锚：约 `0.70236`
   - 风险：预测与已知第三名 V1 完全相同，仅 CSV 序列化 SHA 不同。

2. `submission_2_upside_v1_v7_equal_rank.csv`
   - 固定 50/50 等权 rank 融合，没有搜索权重。
   - 本地 OOF：`0.70028365`
   - 相对 V1 增益：`+0.00732407`
   - 配对 bootstrap 95% CI：`[0.00335866, 0.01128013]`
   - 风险：V7 包缺少完整训练源码、逐 seed 和完整 shuffled 产物；公开榜没有锚点。
     只能作为愿意承担更高风险的第二枪，公开预估约 `0.70–0.71`。

## 明确不采用

- MiniMax `0.55*V5 + 0.45*V3`：权重来自 OOF 网格搜索，违反 R07；且等价于
  `0.775*V5 + 0.225*V1`，进一步放大已导致 V3 公开掉分的 V5/TE 半边。
- Grok：诚实 OOF 仅 `0.66165`。

完整机器证据与 SHA：`FINAL_CANDIDATES.json`。

## 独立复核（已通过）

- 两文件格式与 `submit_sample.csv` 对齐：10000 行、`id` 一致、预测 ∈[0,1]、10000 个唯一值。
- `submission_1` 预测与已知第三名 `submission_v1_3rd_repro.csv` 逐值相同（maxabs=0）；CSV 序列化 SHA 不同（`3ce0e65e…` vs `49ece575…`）。
- `submission_2` 可由 `scripts/build_final_candidates.py` 字节级复现；固定等权 rank，无 OOF 权重搜索。
- `SUBMIT_GATE_0.72` 仍为 FAIL：公开可信证据约 0.70–0.71，未达 0.72 门禁。
