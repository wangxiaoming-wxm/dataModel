#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from insurance_claim.b6pro_fusion import nested_select_rule, apply_rule

B7_FLOOR=0.7027049552615718
GATE=0.71
CLOSEST=float(json.load(open('artifacts/b6pro_long_best/metrics.json'))['nested_oof_auc'])

train=pd.read_csv('train.csv'); test=pd.read_csv('test.csv'); sample=pd.read_csv('submit_sample.csv')
y=train['label'].astype(int); features=train.drop(columns=['label'])
days=features['days'].to_numpy(float); long=days>=3000
region=features['region'].astype(str).to_numpy()
region_te=test['region'].astype(str).to_numpy()
days_te=test['days'].to_numpy(float)

def enrich(df):
    out=df.copy()
    out['log_days']=np.log1p(pd.to_numeric(out['days'],errors='coerce').clip(lower=0))
    cond=pd.to_numeric(out['condition'],errors='coerce')
    d=pd.to_numeric(out['days'],errors='coerce')
    out['ratio']=cond/(d.abs()+1)
    out['days_x_invcond']=d/(cond.abs()+1)
    out['car']=out['source'].astype(str).str.extract(r'(CAR_\d+)',expand=False).fillna('__NA__')
    out['t3_sfx']=out['t3'].astype(str).str.extract(r'([A-Za-z])$',expand=False).fillna('__NONE__')
    db=pd.qcut(d.rank(method='first'),5,labels=False,duplicates='drop').astype(str)
    out['region_days5']=(out['region'].astype(str)+'|'+db).astype(str)
    cb=pd.qcut(cond.rank(method='first'),5,labels=False,duplicates='drop').astype(str)
    out['days5_cond5']=(db+'|'+cb).astype(str)
    out['car_days5']=(out['car'].astype(str)+'|'+db).astype(str)
    return out

etr=enrich(features); ete=enrich(test)
num_cols=[c for c in etr.columns if pd.api.types.is_numeric_dtype(etr[c])]
cat_cols=[c for c in etr.columns if c not in num_cols]

def prep(tr_idx, va_idx):
    enc=OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)
    Xtr=etr.iloc[tr_idx].copy(); Xva=etr.iloc[va_idx].copy(); Xte=ete.copy()
    Xtr[cat_cols]=enc.fit_transform(Xtr[cat_cols].astype(str))
    Xva[cat_cols]=enc.transform(Xva[cat_cols].astype(str))
    Xte[cat_cols]=enc.transform(Xte[cat_cols].astype(str))
    use=num_cols+cat_cols
    for c in use:
        med=pd.to_numeric(Xtr[c],errors='coerce').median()
        for D in (Xtr,Xva,Xte):
            D[c]=pd.to_numeric(D[c],errors='coerce').fillna(med)
    return Xtr[use].to_numpy(), Xva[use].to_numpy(), Xte[use].to_numpy()

def weights(idx, w_f09d=6.0, w_weak=3.0):
    w=np.ones(len(idx)); r=region[idx]; d=days[idx]; long_i=d>=3000
    w[long_i]*=1.2
    w[np.isin(r,['9685','908d','fafc','f167','ab86'])&long_i]=w_weak
    w[(r=='f09d')&long_i]=w_f09d
    return w

b7=np.load('reference/b7_closest/predictions.npz')
fr=np.load('artifacts/b6pro_frozen/predictions.npz')
cur=np.load('artifacts/b6pro_long_best/predictions.npz')
nest=np.load('artifacts/b6pro_nest_div/predictions.npz')

seeds=[2026,2027,2028,2029]
oof_acc=np.zeros(len(y)); te_acc=np.zeros(len(test))
for seed in seeds:
    oof=np.zeros(len(y)); pte=np.zeros(len(test))
    for fold,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(features,y)):
        Xtr,Xva,Xte=prep(tr,va)
        w=weights(tr)
        model=LGBMClassifier(n_estimators=800,learning_rate=0.04,num_leaves=40,subsample=0.85,
            colsample_bytree=0.8,reg_lambda=5,min_child_samples=40,
            random_state=seed+fold,verbosity=-1,n_jobs=3)
        model.fit(Xtr,y.iloc[tr],sample_weight=w)
        oof[va]=model.predict_proba(Xva)[:,1]
        pte+=model.predict_proba(Xte)[:,1]/5
        print(f's{seed} f{fold} {roc_auc_score(y.iloc[va],oof[va]):.5f}',flush=True)
    f09=(region=='f09d')&long
    print(f's{seed} OOF={roc_auc_score(y,oof):.6f} f09d={roc_auc_score(y.to_numpy()[f09],oof[f09]):.5f}',flush=True)
    oof_acc+=oof; te_acc+=pte
oof_l=oof_acc/len(seeds); te_l=te_acc/len(seeds)
f09=(region=='f09d')&long
print('pooled', roc_auc_score(y,oof_l), 'f09d', roc_auc_score(y.to_numpy()[f09],oof_l[f09]), flush=True)

base=nest['oof']; tbase=nest['test']
variants={}
for a in [0.1,0.15,0.2,0.25,0.3,0.4]:
    arm=base.copy(); tarm=tbase.copy()
    m=f09; m_te=(region_te=='f09d')&(days_te>=3000)
    arm[m]=(1-a)*base[m]+a*oof_l[m]; tarm[m_te]=(1-a)*tbase[m_te]+a*te_l[m_te]
    variants[f'f09d_a{a}']=(arm,tarm)
    print(f'f09d_a{a}', roc_auc_score(y,arm), flush=True)
variants['raw']=(oof_l,te_l)
variants['mean_nest']=(0.5*(base+oof_l),0.5*(tbase+te_l))
for a in [0.15,0.25,0.35]:
    arm=base.copy(); tarm=tbase.copy()
    weak=['f09d','9685','908d','fafc']
    m=np.isin(region,weak)&long; m_te=np.isin(region_te,weak)&(days_te>=3000)
    arm[m]=(1-a)*base[m]+a*oof_l[m]; tarm[m_te]=(1-a)*tbase[m_te]+a*te_l[m_te]
    variants[f'weak_a{a}']=(arm,tarm)

# also seq patch like before but with new helper
arm=base.copy(); tarm=tbase.copy()
for reg,a0 in [('f09d',0.25),('9685',0.1),('908d',0.1),('fafc',0.25)]:
    m=(region==reg)&long; m_te=(region_te==reg)&(days_te>=3000)
    # nested pick a
    best_a,best_auc=a0,-1
    for a in [0.05,0.1,0.15,0.2,0.25,0.3,0.4]:
        oof=np.zeros(m.sum()); idx=np.where(m)[0]
        for tr,va in StratifiedKFold(5,shuffle=True,random_state=0).split(np.zeros(len(idx)),y.to_numpy()[idx]):
            oof[va]=(1-a)*base[idx[va]]+a*oof_l[idx[va]]
        auc=roc_auc_score(y.to_numpy()[idx],oof)
        if auc>best_auc: best_auc,best_a=auc,a
    arm[m]=(1-best_a)*arm[m]+best_a*oof_l[m]
    tarm[m_te]=(1-best_a)*tarm[m_te]+best_a*te_l[m_te]
    print(f'seq {reg} a={best_a} overall={roc_auc_score(y,arm):.6f}', flush=True)
variants['seq_lgbw']= (arm,tarm)

results={}; best=None
for name,(oa,ta) in variants.items():
    direct=float(roc_auc_score(y,oa))
    for tag,arms,te_arms in [
        (f'direct_{name}',[oa],[ta]),
        (f'b7+{name}',[b7['gap'],b7['gap_bag'],b7['plus'],oa],[fr['test_gap'],fr['test_gap_bag'],fr['test_plus'],ta]),
        (f'nest+{name}',[b7['gap'],b7['gap_bag'],b7['plus'],base,oa],[fr['test_gap'],fr['test_gap_bag'],fr['test_plus'],tbase,ta]),
    ]:
        if len(arms)==1:
            res={'nested_oof_auc':direct,'nested_oof':oa,'selected_rule':'mean'}
        else:
            res=nested_select_rule(y.to_numpy(),arms)
        results[tag]=float(res['nested_oof_auc'])
        if best is None or res['nested_oof_auc']>best[0]:
            best=(res['nested_oof_auc'],tag,res,te_arms)
deliver,tag,res,te_arms=best
deliver_oof=res['nested_oof']
deliver_test=apply_rule(res['selected_rule'], te_arms) if len(te_arms)>1 else te_arms[0]
promoted=deliver>CLOSEST+1e-12
out=Path('artifacts/b6pro_lgb_weakw'); out.mkdir(parents=True,exist_ok=True)
np.savez_compressed(out/'predictions.npz',y=y.to_numpy(),oof=deliver_oof,test=deliver_test,oof_l=oof_l,te_l=te_l)
lab=[c for c in sample.columns if c!='id'][0]
sub=sample.copy(); sub[lab]=deliver_test; sub.to_csv(out/'submission_b6pro.csv',index=False)
if promoted:
    dest=Path('artifacts/b6pro_long_best')
    np.savez_compressed(dest/'predictions.npz',y=y.to_numpy(),oof=deliver_oof,test=deliver_test)
    sub.to_csv(dest/'submission_b6pro.csv',index=False)
    sub.to_csv('submissions/b6pro_closest/submission_b6pro.csv',index=False)
    (dest/'metrics.json').write_text(json.dumps({'experiment_id':'b6pro_long_best','spec':tag,'nested_oof_auc':deliver,'baseline_max3':B7_FLOOR,'gate_0_71':deliver>=GATE,'gap_to_0_71':GATE-deliver,'source':'b6pro_lgb_weakw'},indent=2))
metrics={'best':tag,'nested':deliver,'lgb':float(roc_auc_score(y,oof_l)),'f09d':float(roc_auc_score(y.to_numpy()[f09],oof_l[f09])),'promoted':promoted,'gate':deliver>=GATE,'top':sorted(results.items(),key=lambda kv:-kv[1])[:12]}
(out/'metrics.json').write_text(json.dumps(metrics,indent=2))
print(json.dumps(metrics,indent=2), flush=True)
print(f"GATE={'PASS' if deliver>=GATE else 'FAIL'} nested={deliver:.8f} promoted={promoted}", flush=True)
