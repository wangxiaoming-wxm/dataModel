# B6pro 冻结说明（closest honest）

## 分数

- **nested_oof_auc = 0.7027049552615718**
- selected_rule = `max`
- gate_0_705 = false；gate_0_715 = false

## 复现

```bash
PYTHONPATH=src python3 - <<'PY'
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from insurance_claim.b6pro_fusion import nested_select_rule
y = pd.read_csv('train.csv')['label'].to_numpy()
main = np.load('artifacts/b6pro_main/predictions.npz')  # or arms/main_8seed.npz
plus = np.load('reference/v10/oof_plus_h2_10.npz')
r = nested_select_rule(y, [main['oof_gap'], main['oof_gap_bag'], plus['oof']])
assert abs(r['nested_oof_auc'] - 0.7027049552615718) < 1e-12
print(r['nested_oof_auc'], r['selected_rule'])
PY
```

冻结产物：`artifacts/b6pro_frozen/{metrics.json,predictions.npz,FREEZE.json,submission_b6pro.csv}`。
