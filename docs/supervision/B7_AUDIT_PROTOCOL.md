# 车险索赔 AUC · B7 独立监督协议（IA-AUC710-B7-v1）

> **角色**：独立监督者 / 复核官。不参与写抬分代码。  
> **对象**：分支 `cursor/b7-push-auc071-a5f5` 冲刺诚实本地 OOF **≥ 0.71**。  
> **冻结**：B6 closest honest **0.69897470**；B5 **0.69817454**。B7 **不得篡改** B5/B6 冻结目录。  
> **用户约束**：仅无过拟合/作弊时可交付 0.71；否则只报 closest honest。

继承 `B6_AUDIT_PROTOCOL.md` 全部红线；下列为相对 B6/V10 的加严。

---

## 1. 分数口径

| 口径 | 要求 |
|---|---|
| **主报** | **nested_oof_auc**（嵌套选融合规则） |
| 副报 | full-data 选定规则 pooled；各臂 OOF；公开榜另列 |
| 门禁 | `nested_oof_auc ≥ 0.71` 才可主张达标 |
| seeds | 主臂 ≥8 或 plus 臂按预注册（V10 plus 为 4×10fold，须披露） |

## 2. 融合红线

- 规则集合须**开跑前**写入计划/`protocol_declaration`（允许 V10 六规则）  
- 允许嵌套折选离散规则；**禁止**连续权重 / 看完 full OOF 再发明新规则  
- 若 ≥0.71 **仅**在 `max` 上成立：须 nested 稳定选中 + 披露；监督者可标 **CONDITIONAL**（叙事绑定 max）  
- `shuffled_plus` 后 max 须明显崩盘（<0.66）

## 3. 硬红线（一票否决）

test 标签 / 伪标签 / 全局 TE / 旧数据 / OOF 连续搜权 / 篡改 B5·B6 冻结 / 伪造 nested

## 4. 裁决

- **PASS**：nested≥0.71 且红线全过（若依赖 max 则 CONDITIONAL PASS，须披露）  
- **REJECT**：nested<0.71 或红线命中；可认证 `closest_honest_nested`

## 5. 开跑前检查

```bash
# B6 冻结 submission sha 仍在 submissions/b6_frozen/
# B5 冻结仍在 artifacts/b5_frozen/
```
