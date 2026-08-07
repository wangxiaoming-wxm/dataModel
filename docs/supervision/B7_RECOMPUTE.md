# B7 独立复算说明

外人无需作者本机 OOF，可用仓库内已提交产物复算。

## 必要文件（已入库）

| 路径 | 内容 |
|---|---|
| `artifacts/b6_frozen/predictions.npz` | B6 `oof_gap` / `oof_gap_bag` / `test_*` / `y` |
| `reference/v10/oof_plus_h2_10.npz` | V10 plus OOF（含分 seed） |
| `reference/v10/test_plus_h2_10.npy` | V10 plus test |
| `artifacts/b7_closest/predictions.npz` | closest 打包：`oof/test/y/gap/gap_bag/plus` |
| `artifacts/b7_fuse0_b6/predictions.npz` | pair fuse0 对照 |

`.gitignore` 对上述目录做了 `npz` 白名单例外。

## 一键复算

```bash
PYTHONPATH=src python3 scripts/b7_recompute_closest.py
# → artifacts/b7_audit/recompute_closest.json
# pass_recompute_lt_1e-8 应为 true
```

期望：

- closest `max(gap,gap_bag,plus)` = **0.7027049552615718**
- fuse0 pair nested = **0.7022093156561012**
- B6 equal = **0.6989746962571622**
