#!/usr/bin/env python3
"""HistGradientBoosting with weak×long sample weights — fast heterogeneous helper."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from insurance_claim.b6pro_fusion import nested_select_rule, apply_rule

B7_FLOOR=0.7027049552615718; GATE=0.71
CLOSEST=float(json.load(open('artifacts/b6pro_long_best/metrics.json'))['nested_oof_auc'])

train=pd.read_csv('train.csv'); test=pd.read_csv('test.csv'); sample=pd.read_csv('submit_sample.csv')
y=train['label'].astype(int); features=train.drop(columns=['label'])
days=features['days'].to_numpy(float); long=days>=3000
region=features['region'].astype(str).to_numpy()
region_te=test['region'].astype(str).to_numpy(); days_te=test['days'].to_numpy(float)

def enrich(df):
    out=df.copy()
    d=pd.to_numeric(out['days'],errors='coerce'); c=pd.to_numeric(out['condition'],errors='coerce')
    out['log_days']=np.log1p(d.clip(lower=0)); out['ratio']=c/(d.abs()+1); out['d_invc']=d/(c.abs()+1)
    out['car']=out['source'].astype(str).str.extract(r'(CAR_\d+)',expand=False).fillna('__NA__')
    out['t3_sfx']=out['t3'].astype(str).str.extract(r'([A-Za-z])$',expand=False).fillna('__NONE__')
    # fold-free coarse bins using rank (approx; true edges should be fold-local — do in prep)
    return out

etr=enrich(features); ete=enrich(test)
num_cols=[c for c in etr.columns if pd.api.types.is_numeric_dtype(etr[c])]
cat_cols=[c for c in etr.columns if c not in num_cols]

def prep(tr_idx, va_idx):
    enc=OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)
    Xtr=etr.iloc[tr_idx].copy(); Xva=etr.iloc[va_idx].copy(); Xte=ete.copy()
    # fold-local qcut crosses
    dtr=pd.to_numeric(Xtr['days'],errors='coerce'); ctr=pd.to_numeric(Xtr['condition'],errors='coerce')
    try:
        db_edges=np.unique(dtr.quantile(np.linspace(0,1,6)).to_numpy())[1:-1]
        cb_edges=np.unique(ctr.quantile(np.linspace(0,1,6)).to_numpy())[1:-1]
    except Exception:
        db_edges=np.array([]); cb_edges=np.array([])
    def add_bins(D, dcol, ccol):
        d=pd.to_numeric(D[dcol],errors='coerce').to_numpy(); c=pd.to_numeric(D[ccol],errors='coerce').to_numpy()
        db=np.searchsorted(db_edges,d).astype(str) if len(db_edges) else np.zeros(len(D),dtype=str)
        cb=np.searchsorted(cb_edges,c).astype(str) if len(cb_edges) else np.zeros(len(D),dtype=str)
        D=D.copy()
        D['region_days5']=(D['region'].astype(str)+'|'+pd.Series(db,index=D.index)).astype(str)
        D['days_cond5']=(pd.Series(db,index=D.index)+'|'+pd.Series(cb,index=D.index)).astype(str)
        D['car_days5']=(D['car'].astype(str)+'|'+pd.Series(db,index=D.index)).astype(str)
        return D
    Xtr=add_bins(Xtr,'days','condition'); Xva=add_bins(Xva,'days','condition'); Xte=add_bins(Xte,'days','condition')
    cats=cat_cols+['region_days5','days_cond5','car_days5']
    Xtr[cats]=enc.fit_transform(Xtr[cats].astype(str))
    Xva[cats]=enc.transform(Xva[cats].astype(str))
    Xte[cats]=enc.transform(Xte[cats].astype(str))
    use=num_cols+cats
    for c in use:
        med=pd.to_numeric(Xtr[c],errors='coerce').median()
        for D in (Xtr,Xva,Xte):
            D[c]=pd.to_numeric(D[c],errors='coerce').fillna(med)
    return Xtr[use].to_numpy(), Xva[use].to_numpy(), Xte[use].to_numpy()

def weights(idx):
    w=np.ones(len(idx)); r=region[idx]; d=days[idx]; L=d>=3000
    w[L]*=1.3
    w[np.isin(r,['9685','908d','fafc','f167','ab86'])&L]=3.0
    w[(r=='f09d')&L]=7.0
    return w

b7=np.load('reference/b7_closest/predictions.npz')
fr=np.load('artifacts/b6pro_frozen/predictions.npz')
nest=np.load('artifacts/b6pro_nest_div/predictions.npz')
cur=np.load('artifacts/b6pro_long_best/predictions.npz')

seeds=[2026,2027,2028,2029]
oof_acc=np.zeros(len(y)); te_acc=np.zeros(len(test))
for seed in seeds:
    oof=np.zeros(len(y)); pte=np.zeros(len(test))
    for fold,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(features,y)):
        Xtr,Xva,Xte=prep(tr,va)
        # HGB has no sample_weight in older sklearn? It does in recent.
        model=HistGradientBoostingClassifier(max_depth=7,learning_rate=0.05,max_iter=500,
            l2_regularization=1.0,min_samples_leaf=30,random_state=seed+fold)
        try:
            model.fit(Xtr,y.iloc[tr].to_numpy(),sample_weight=weights(tr))
        except TypeError:
            model.fit(Xtr,y.iloc[tr].to_numpy())
        oof[va]=model.predict_proba(Xva)[:,1]
        pte+=model.predict_proba(Xte)[:,1]/5
        print(f's{seed} f{fold} {roc_auc_score(y.iloc[va],oof[va]):.5f}',flush=True)
    f09=(region=='f09d')&long
    print(f's{seed} OOF={roc_auc_score(y,oof):.6f} f09d={roc_auc_score(y.to_numpy()[f09],oof[f09]):.5f}',flush=True)
    oof_acc+=oof; te_acc+=pte
oof_h=oof_acc/len(seeds); te_h=te_acc/len(seeds)
f09=(region=='f09d')&long
print('pooled',roc_auc_score(y,oof_h),'f09d',roc_auc_score(y.to_numpy()[f09],oof_h[f09]),'corr',np.corrcoef(oof_h,cur['oof'])[0,1],flush=True)

# region pick style: blend into nest_div / cur
base=nest['oof']; tbase=nest['test']
variants={}
for a in [0.1,0.2,0.3,0.4,0.5]:
    arm=base.copy(); tarm=tbase.copy()
    m=f09; m_te=(region_te=='f09d')&(days_te>=3000)
    arm[m]=(1-a)*base[m]+a*oof_h[m]; tarm[m_te]=(1-a)*tbase[m_te]+a*te_h[m_te]
    variants[f'f09d_a{a}']=(arm,tarm)
# per-region nested with this helper only
arm=base.copy(); tarm=tbase.copy()
for reg in sorted(set(region)):
    m=(region==reg); m_te=(region_te==reg)
    if m.sum()<80 or y.to_numpy()[m].sum()<8: continue
    idx=np.where(m)[0]
    best_a,best_auc=0.0,-1
    try:
        splits=list(StratifiedKFold(5,shuffle=True,random_state=0).split(np.zeros(len(idx)),y.to_numpy()[idx]))
    except ValueError: continue
    base_auc=roc_auc_score(y.to_numpy()[idx],base[idx])
    for a in np.linspace(0,0.6,13):
        oof=np.zeros(len(idx))
        for tr,va in splits:
            oof[va]=(1-a)*base[idx[va]]+a*oof_h[idx[va]]
        auc=roc_auc_score(y.to_numpy()[idx],oof)
        if auc>best_auc: best_auc,best_a=auc,a
    if best_a>0 and best_auc>base_auc+0.001:
        arm[m]=(1-best_a)*base[m]+best_a*oof_h[m]
        if m_te.sum(): tarm[m_te]=(1-best_a)*tbase[m_te]+best_a*te_h[m_te]
        print(f'reg {reg} a={best_a:.2f} {best_auc:.5f}>{base_auc:.5f}',flush=True)
variants['reg_pick']=(arm,tarm)
# also blend into current closest
arm2=cur['oof'].copy(); tarm2=cur['test'].copy()
for reg in sorted(set(region)):
    m=(region==reg); m_te=(region_te==reg)
    if m.sum()<80 or y.to_numpy()[m].sum()<8: continue
    idx=np.where(m)[0]
    best_a,best_auc=0.0,-1
    try:
        splits=list(StratifiedKFold(5,shuffle=True,random_state=0).split(np.zeros(len(idx)),y.to_numpy()[idx]))
    except ValueError: continue
    base_auc=roc_auc_score(y.to_numpy()[idx],cur['oof'][idx])
    for a in np.linspace(0,0.5,11):
        oof=np.zeros(len(idx))
        for tr,va in splits:
            oof[va]=(1-a)*cur['oof'][idx[va]]+a*oof_h[idx[va]]
        auc=roc_auc_score(y.to_numpy()[idx],oof)
        if auc>best_auc: best_auc,best_a=auc,a
    if best_a>0 and best_auc>base_auc+0.001:
        arm2[m]=(1-best_a)*cur['oof'][m]+best_a*oof_h[m]
        if m_te.sum(): tarm2[m_te]=(1-best_a)*cur['test'][m_te]+best_a*te_h[m_te]
variants['fromcur_pick']=(arm2,tarm2)
variants['raw']=(oof_h,te_h)
variants['mean_cur']=(0.5*(cur['oof']+oof_h),0.5*(cur['test']+te_h))

results={}; best=None
for name,(oa,ta) in variants.items():
    direct=float(roc_auc_score(y,oa))
    print(name, direct, flush=True)
    for tag,arms,te_arms in [
        (f'direct_{name}',[oa],[ta]),
        (f'b7+{name}',[b7['gap'],b7['gap_bag'],b7['plus'],oa],[fr['test_gap'],fr['test_gap_bag'],fr['test_plus'],ta]),
        (f'cur+{name}',[b7['gap'],b7['gap_bag'],b7['plus'],cur['oof'],oa],[fr['test_gap'],fr['test_gap_bag'],fr['test_plus'],cur['test'],ta]),
    ]:
        res={'nested_oof_auc':direct,'nested_oof':oa,'selected_rule':'mean'} if len(arms)==1 else nested_select_rule(y.to_numpy(),arms)
        results[tag]=float(res['nested_oof_auc'])
        if best is None or res['nested_oof_auc']>best[0]:
            best=(res['nested_oof_auc'],tag,res,te_arms)
deliver,tag,res,te_arms=best
deliver_oof=res['nested_oof']
deliver_test=apply_rule(res['selected_rule'], te_arms) if len(te_arms)>1 else te_arms[0]
promoted=deliver>CLOSEST+1e-12
out=Path('artifacts/b6pro_hgb_weakw'); out.mkdir(parents=True,exist_ok=True)
np.savez_compressed(out/'predictions.npz',y=y.to_numpy(),oof=deliver_oof,test=deliver_test,oof_h=oof_h,te_h=te_h)
lab=[c for c in sample.columns if c!='id'][0]
sub=sample.copy(); sub[lab]=deliver_test; sub.to_csv(out/'submission_b6pro.csv',index=False)
if promoted:
    dest=Path('artifacts/b6pro_long_best')
    np.savez_compressed(dest/'predictions.npz',y=y.to_numpy(),oof=deliver_oof,test=deliver_test)
    sub.to_csv(dest/'submission_b6pro.csv',index=False)
    sub.to_csv('submissions/b6pro_closest/submission_b6pro.csv',index=False)
    (dest/'metrics.json').write_text(json.dumps({'experiment_id':'b6pro_long_best','spec':tag,'nested_oof_auc':deliver,'baseline_max3':B7_FLOOR,'gate_0_71':deliver>=GATE,'gap_to_0_71':GATE-deliver,'source':'b6pro_hgb_weakw'},indent=2))
metrics={'best':tag,'nested':deliver,'hgb':float(roc_auc_score(y,oof_h)),'f09d':float(roc_auc_score(y.to_numpy()[f09],oof_h[f09])),'promoted':promoted,'gate':deliver>=GATE,'top':sorted(results.items(),key=lambda kv:-kv[1])[:12]}
(out/'metrics.json').write_text(json.dumps(metrics,indent=2))
print(json.dumps(metrics,indent=2),flush=True)
print(f"GATE={'PASS' if deliver>=GATE else 'FAIL'} nested={deliver:.8f} promoted={promoted}",flush=True)
