# 预测数组包（复现 final_candidates 用）

来源：`minimax-m3=team4_push_top3_blends.zip`（main 上的外部交付包）。

| 文件 | 说明 |
|---|---|
| `oof_v1_3rd.npy` / `test_v1_3rd.npy` | V1（第三名语义 4-seed）OOF/test |
| `oof_v7_lgbm.npy` / `test_v7_lgbm.npy` | V7 LightGBM bag；MiniMax README 称其训，**交付包未含训练脚本** |
| `y.npy` / `test_id.npy` | 标签与 test id，供校验 |

复现最终候选：

```bash
python3 scripts/build_final_candidates.py \
  --data-dir . \
  --package-dir artifacts/pred_bundle \
  --output-dir final_candidates
```

说明：仓库内**没有** `03_train_v7_lgbm.py`。MiniMax 包仅提供 `.npy` 与 blend 脚本，未提供 V7 训练源码。
