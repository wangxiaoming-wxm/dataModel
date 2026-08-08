# B7 / B6 训练说明

## 权威分数怎么来的

B7 closest 本地 AUC **不是**在本机临时搜参得到的，而是：

1. B6：`gap` + `gap_bag`，各 8 个 seed、5-fold OOF，等权得到臂预测  
2. V10 plus：10-fold × 4 seed 池化 OOF  
3. 融合：`elementwise max` 三臂  

冻结文件：

- `artifacts/b6_frozen/predictions.npz`（键：`oof_gap`,`oof_gap_bag`,`test_gap`,`test_gap_bag`,`y`,…）
- `reference/v10/oof_plus_h2_10.npz` / `test_plus_h2_10.npy`
- `artifacts/b7_closest/predictions.npz`（键：`oof`,`test`,`y`,`gap`,`gap_bag`,`plus`）

验证：`PYTHONPATH=src python3 scripts/b7_recompute_closest.py`

## 可选：从零重训 B6 臂

```bash
ln -sfn data/train.csv train.csv
ln -sfn data/test.csv test.csv
ln -sfn data/submit_sample.csv submit_sample.csv
PYTHONPATH=src python3 -m insurance_claim.train_b6 \
  --arms gap gap_bag --fuse-arms gap gap_bag \
  --output-dir artifacts/b6_retrain
```

依赖：`feature_blocks` + `train_b5_focus`（B5 特征底座）+ `b6_gap_features`。

## 可选：重训 plus

```bash
PYTHONPATH=src python3 -m insurance_claim.train_b7_plus \
  --seeds 2026 2027 2028 2029 --folds 10 \
  --output-dir artifacts/b7_plus_retrain
```

H2 参数见 `train_b7_plus.py` 内 `PARAMS_H2`（2500 iter / lr0.02 / depth7 / l2=20）。

## 融合出提交

```bash
PYTHONPATH=src python3 scripts/fuse_b7_closest.py
# 或指定重训 OOF：
PYTHONPATH=src python3 scripts/fuse_b7_closest.py \
  --b6 artifacts/b6_retrain/predictions.npz \
  --plus-oof artifacts/b7_plus_retrain/predictions.npz
```

## 协议要点（训练时勿违反）

- FE 必须在折内 fit  
- 不要用全量数据做 TE 再 CV  
- 不要连续搜 OOF 融合权重冒充 nested  
- 不要用 test 标签 / 伪标签
