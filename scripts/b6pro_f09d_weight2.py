#!/usr/bin/env python3
"""Moderate f09d×long weight keepx (w=2.5) — avoid over-weight collapse."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from insurance_claim.b6pro_fusion import nested_select_rule, apply_rule
from insurance_claim.b6pro_long_features import build_long_keepx
from insurance_claim.train_b6 import PARAMS_GAP_BAG

B7_FLOOR=0.7027049552615718; GATE=0.71
CLOSEST=float(json.load(open('artifacts/b6pro_long_best/metrics.json'))['nested_oof_auc'])
PARAMS={**PARAMS_GAP_BAG,'thread_count':3,'iterations':2800,'od_wait':140,'depth':8,'l2_leaf_reg':8}

train=pd.read_csv('train.csv'); test=pd.read_csv('test.csv'); sample=pd.read_csv('submit_sample.csv')
y=train['label'].astype(int); features=train.drop(columns=['label'])
days=features['days'].to_numpy(float); long=days>=3000
region=features['region'].astype(str).to_numpy()
region_te=test['region'].astype(str).to_numpy(); days_te=test['days'].to_numpy(float)

def make_w(X, wf=2.5, ww=1.8):
    r=X['region'].astype(str).to_numpy(); d=X['days'].to_numpy(float)
    w=np.ones(len(X)); L=d>=3000
    w[L]*=1.15
    w[np.isin(r,['9685','908d','fafc','f167','ab86'])&L]=ww
    w[(r=='f09d')&L]=wf
    return w

b7=np.load('reference/b7_closest/predictions.npz'); fr=np.load('artifacts/b6pro_frozen/predictions.npz')
cur=np.load('artifacts/b6pro_long_best/predictions.npz')
nest=np.load('artifacts/b6pro_nest_div/predictions.npz')
pick=np.load('artifacts/b6pro_region_pick/predictions.npz')
blend3=np.load('artifacts/b6pro_region_blend3/predictions.npz')

seeds=[2026,2027,2028,2029]
oof_acc=np.zeros(len(y)); te_acc=np.zeros(len(test))
for seed in seeds:
    oof=np.zeros(len(y)); pte=np.zeros(len(test))
    for fold,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(features,y)):
        trd,vad,ted,cats=build_long_keepx(features.iloc[tr].reset_index(drop=True), features.iloc[va].reset_index(drop=True), test.copy())
        w=make_w(features.iloc[tr].reset_index(drop=True))
        model=CatBoostClassifier(**{**PARAMS,'random_seed':seed+fold})
        model.fit(trd,y.iloc[tr],sample_weight=w,eval_set=(vad,y.iloc[va]),cat_features=cats,use_best_model=True)
        oof[va]=model.predict_proba(vad)[:,1]; pte+=model.predict_proba(ted)[:,1]/5
        print(f's{seed} f{fold} {roc_auc_score(y.iloc[va],oof[va]):.5f}',flush=True)
    f09=(region=='f09d')&long
    print(f's{seed} OOF={roc_auc_score(y,oof):.6f} f09d={roc_auc_score(y.to_numpy()[f09],oof[f09]):.5f} long={roc_auc_score(y.to_numpy()[long],oof[long]):.5f}',flush=True)
    oof_acc+=oof; te_acc+=pte
oof_w=oof_acc/len(seeds); te_w=te_acc/len(seeds)
f09=(region=='f09d')&long
print('pooled',roc_auc_score(y,oof_w),'f09d',roc_auc_score(y.to_numpy()[f09],oof_w[f09]),'corr_cur',np.corrcoef(oof_w,cur['oof'])[0,1],flush=True)

# honest nest alpha into cur / pick / blend3 on f09d and all-region
def honest_alpha(base,help_,tbase,thelp,mask=None,alphas=np.linspace(0,0.5,11)):
    if mask is None:
        idx=np.arange(len(y))
    else:
        idx=np.where(mask)[0]
    oof=base.copy(); fold_as=[]
    # nested on full for global, or on mask then apply to mask
    if mask is None:
        for otr,ova in StratifiedKFold(5,shuffle=True,random_state=0).split(np.zeros(len(y)),y):
            best_a,best_auc=0.0,-1
            for a in alphas:
                auc=roc_auc_score(y[otr],(1-a)*base[otr]+a*help_[otr])
                if auc>best_auc: best_auc,best_a=auc,a
            fold_as.append(best_a); oof[ova]=(1-best_a)*base[ova]+best_a*help_[ova]
        a_star=float(np.median(fold_as))
        te=(1-a_star)*tbase+a_star*thelp
    else:
        for otr,ova in StratifiedKFold(5,shuffle=True,random_state=0).split(np.zeros(len(idx)),y[idx]):
            best_a,best_auc=0.0,-1
            for a in alphas:
                auc=roc_auc_score(y[idx[otr]],(1-a)*base[idx[otr]]+a*help_[idx[otr]])
                if auc>best_auc: best_auc,best_a=auc,a
            fold_as.append(best_a)
            oof[idx[ova]]=(1-best_a)*base[idx[ova]]+best_a*help_[idx[ova]]
        a_star=float(np.median(fold_as))
        # test mask approx: same regions
        # rebuild test mask from train mask definition externally
        te=tbase.copy()
        # caller patches test
    return oof, te if mask is None else None, a_star, fold_as

variants={}
for bname,bo,bt in [('cur',cur['oof'],cur['test']),('pick',pick['oof'],pick['test']),('blend3',blend3['oof'],blend3['test']),('nest',nest['oof'],nest['test'])]:
    oof,te,a,fas=honest_alpha(bo,oof_w,bt,te_w,mask=None)
    variants[f'global_{bname}_a{a}']=(oof,te)
    print(f'global_{bname}',roc_auc_score(y,oof),a,fas,flush=True)
    # f09d only patch
    oof2=bo.copy(); te2=bt.copy()
    m=f09; m_te=(region_te=='f09d')&(days_te>=3000)
    idx=np.where(m)[0]
    fold_as=[]; oof_m=np.zeros(len(idx))
    for otr,ova in StratifiedKFold(5,shuffle=True,random_state=0).split(np.zeros(len(idx)),y.to_numpy()[idx]):
        best_a,best_auc=0.0,-1
        for a in np.linspace(0,0.6,13):
            auc=roc_auc_score(y.to_numpy()[idx[otr]],(1-a)*bo[idx[otr]]+a*oof_w[idx[otr]])
            if auc>best_auc: best_auc,best_a=auc,a
        fold_as.append(best_a)
        oof_m[ova]=(1-best_a)*bo[idx[ova]]+best_a*oof_w[idx[ova]]
    a_star=float(np.median(fold_as))
    oof2[idx]=oof_m  # use nested oof
    # for consistency also set with a_star full on mask for test
    oof2[m]=(1-a_star)*bo[m]+a_star*oof_w[m]  # overwrite with a_star for deliver simplicity — better keep nested
    oof2[idx]=oof_m
    te2[m_te]=(1-a_star)*bt[m_te]+a_star*te_w[m_te]
    variants[f'f09d_{bname}_a{a_star}']=(oof2,te2)
    print(f'f09d_{bname}',roc_auc_score(y,oof2),a_star,'f09d',roc_auc_score(y.to_numpy()[m],oof2[m]),flush=True)

variants['raw']=(oof_w,te_w)
best=None; results={}
for name,(oa,ta) in variants.items():
    direct=float(roc_auc_score(y,oa))
    for tag,arms,tes in [
        (f'direct_{name}',[oa],[ta]),
        (f'b7+{name}',[b7['gap'],b7['gap_bag'],b7['plus'],oa],[fr['test_gap'],fr['test_gap_bag'],fr['test_plus'],ta]),
        (f'cur+{name}',[b7['gap'],b7['gap_bag'],b7['plus'],cur['oof'],oa],[fr['test_gap'],fr['test_gap_bag'],fr['test_plus'],cur['test'],ta]),
    ]:
        res={'nested_oof_auc':direct,'nested_oof':oa,'selected_rule':'mean'} if len(arms)==1 else nested_select_rule(y.to_numpy(),arms)
        results[tag]=float(res['nested_oof_auc'])
        if best is None or res['nested_oof_auc']>best[0]:
            best=(res['nested_oof_auc'],tag,res,tes if len(arms)>1 else [ta])

deliver,tag,res,tes=best
deliver_oof=res['nested_oof']
deliver_te=apply_rule(res['selected_rule'],tes) if len(tes)>1 else tes[0]
promoted=deliver>CLOSEST+1e-12
out=Path('artifacts/b6pro_f09d_weight2'); out.mkdir(parents=True,exist_ok=True)
np.savez_compressed(out/'predictions.npz',y=y.to_numpy(),oof=deliver_oof,test=deliver_te,oof_w=oof_w,te_w=te_w)
lab=[c for c in sample.columns if c!='id'][0]
sub=sample.copy(); sub[lab]=deliver_te; sub.to_csv(out/'submission_b6pro.csv',index=False)
if promoted:
    dest=Path('artifacts/b6pro_long_best')
    np.savez_compressed(dest/'predictions.npz',y=y.to_numpy(),oof=deliver_oof,test=deliver_te)
    sub.to_csv(dest/'submission_b6pro.csv',index=False)
    sub.to_csv('submissions/b6pro_closest/submission_b6pro.csv',index=False)
    (dest/'metrics.json').write_text(json.dumps({'experiment_id':'b6pro_long_best','spec':tag,'nested_oof_auc':deliver,'baseline_max3':B7_FLOOR,'gate_0_71':deliver>=GATE,'gap_to_0_71':GATE-deliver,'source':'b6pro_f09d_weight2'},indent=2))
metrics={'best':tag,'nested':deliver,'solo':float(roc_auc_score(y,oof_w)),'f09d':float(roc_auc_score(y.to_numpy()[f09],oof_w[f09])),'promoted':promoted,'gate':deliver>=GATE,'top':sorted(results.items(),key=lambda kv:-kv[1])[:12]}
(out/'metrics.json').write_text(json.dumps(metrics,indent=2))
print(json.dumps({k:v for k,v in metrics.items() if k!='top'},indent=2),flush=True)
print('TOP',metrics['top'][:8],flush=True)
print(f"GATE={'PASS' if deliver>=GATE else 'FAIL'} nested={deliver:.8f} promoted={promoted}",flush=True)
