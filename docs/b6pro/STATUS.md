# B6pro 状态

## 公开榜对齐

- B7 closest ≡ 本分支冻结 max3：本地 **0.702704955**
- 用户提交 `submission_b7_closest_honest.csv` → **公开 0.707**（Δ≈+0.0043）
- 目标：诚实本地 nested **≥ 0.71**（继续冲）

## Closest

| 方案 | nested |
|---|---:|
| **max(gap,gap_bag,plus_ref)** | **0.70270496** |
| gap_resid 四臂 | 0.70222 |
| plus_strong 自训≈ref | plus 0.68848；嵌套 0.70145 |
| plus_gate | 0.70270（仍选 max3） |
| stack_raw / mlp | ≤0.7007 |

## 进行中

- gap 10-fold×4seed 重训
- 下一步：弱 region 专模（9685/f09d/fafc/6645）
