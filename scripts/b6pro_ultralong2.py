#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from insurance_claim.b6pro_fusion import nested_select_rule, apply_rule
from insurance_claim.b6pro_long_features import build_long_keepx, build_long_aging
from insurance_claim.train_b6 import PARAMS_GAP_BAG

B7_FLOOR=0.7027049552615718; GATE=0.71
CLOSEST=float(json.load(open('artifacts/b6pro_long_best/metrics.json'))['nested_oof_auc'])
PARAMS={**PARAMS_GAP_BAG,'thread_count':3,'iterations':2200,'od_wait':100,'depth':7}

def main():
    train=pd.read_csv('train.csv'); test=pd.read_csv('test.csv'); sample=pd.read_csv('submit_sample.csv')
    y=train['label'].astype(int); features=train.drop(columns=['label'])
    days=features['days'].to_numpy(float); days_te=test['days'].to_numpy(float)
    ultra=days>=10000; ultra_te=days_te>=10000
    _cur=np.load('artifacts/b6pro_honest_blend/predictions.npz')
    base=_cur['oof'].copy(); tbase=_cur['test'].copy()
    b7=np.load('reference/b7_closest/predictions.npz'); fr=np.load('artifacts/b6pro_frozen/predictions.npz')
    print('base ultra', roc_auc_score(y.to_numpy()[ultra], base[ultra]), flush=True)

    def train_slice(builder, min_days, seeds, name):
        mask=days>=min_days; idx=np.where(mask)[0]
        oof_acc=np.zeros(len(y)); te_acc=np.zeros(len(test))
        for seed in seeds:
            oof=np.zeros(len(y)); pte=np.zeros(len(test))
            Xl=features.iloc[idx].reset_index(drop=True); yl=y.iloc[idx].reset_index(drop=True)
            for fold,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(Xl,yl)):
                gtr,gva=idx[tr],idx[va]
                trd,vad,ted,cats=builder(features.iloc[gtr].reset_index(drop=True), features.iloc[gva].reset_index(drop=True), test.copy())
                model=CatBoostClassifier(**{**PARAMS,'random_seed':seed+fold})
                model.fit(trd,y.iloc[gtr],eval_set=(vad,y.iloc[gva]),cat_features=cats,use_best_model=True)
                oof[gva]=model.predict_proba(vad)[:,1]; pte+=model.predict_proba(ted)[:,1]/5
            print(name,'s',seed,'slice',roc_auc_score(y.to_numpy()[mask],oof[mask]), flush=True)
            oof_acc+=oof; te_acc+=pte
        return oof_acc/len(seeds), te_acc/len(seeds)

    seeds=[2026,2027,2028]
    o7,t7=train_slice(build_long_keepx,7000,seeds,'keepx7k')
    o10,t10=train_slice(build_long_keepx,10000,seeds,'keepx10k')
    o7a,t7a=train_slice(build_long_aging,7000,seeds,'aging7k')

    variants={}
    for tag,oof_s,te_s,mask,mask_te in [
        ('u_kx10',o10,t10,ultra,ultra_te),('u_kx7',o7,t7,ultra,ultra_te),('u_ag7',o7a,t7a,ultra,ultra_te),
        ('7k_kx7',o7,t7,days>=7000,days_te>=7000),('7k_ag7',o7a,t7a,days>=7000,days_te>=7000),
    ]:
        idx=np.where(mask)[0]; oof=base.copy(); oof_m=np.zeros(len(idx)); fold_as=[]
        for otr,ova in StratifiedKFold(5,shuffle=True,random_state=0).split(np.zeros(len(idx)),y.to_numpy()[idx]):
            best_a,best_auc=0.0,-1
            for a in np.linspace(0,1,21):
                auc=roc_auc_score(y.to_numpy()[idx[otr]],(1-a)*base[idx[otr]]+a*oof_s[idx[otr]])
                if auc>best_auc: best_auc,best_a=auc,a
            fold_as.append(best_a); oof_m[ova]=(1-best_a)*base[idx[ova]]+best_a*oof_s[idx[ova]]
        a_star=float(np.median(fold_as)); oof[idx]=oof_m
        te=tbase.copy(); te[mask_te]=(1-a_star)*tbase[mask_te]+a_star*te_s[mask_te]
        variants[tag]=(oof,te)
        print(tag, roc_auc_score(y,oof), 'a', a_star, flush=True)

    best=None; results={}
    for name,(oa,ta) in variants.items():
        direct=float(roc_auc_score(y,oa))
        for tag,arms,tes in [
            (f'direct_{name}',[oa],[ta]),
            (f'b7+{name}',[b7['gap'],b7['gap_bag'],b7['plus'],oa],[fr['test_gap'],fr['test_gap_bag'],fr['test_plus'],ta]),
            (f'cur+{name}',[b7['gap'],b7['gap_bag'],b7['plus'],base,oa],[fr['test_gap'],fr['test_gap_bag'],fr['test_plus'],tbase,ta]),
        ]:
            res={'nested_oof_auc':direct,'nested_oof':oa,'selected_rule':'mean'} if len(arms)==1 else nested_select_rule(y.to_numpy(),arms)
            results[tag]=float(res['nested_oof_auc'])
            if best is None or res['nested_oof_auc']>best[0]:
                best=(res['nested_oof_auc'],tag,res,tes if len(arms)>1 else [ta])
    deliver,tag,res,tes=best
    deliver_oof=res['nested_oof']; deliver_te=apply_rule(res['selected_rule'],tes) if len(tes)>1 else tes[0]
    promoted=deliver>CLOSEST+1e-12
    out=Path('artifacts/b6pro_ultralong2'); out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'predictions.npz',y=y.to_numpy(),oof=deliver_oof,test=deliver_te,o7=o7,o10=o10,o7a=o7a)
    lab=[c for c in sample.columns if c!='id'][0]
    sub=sample.copy(); sub[lab]=deliver_te; sub.to_csv(out/'submission_b6pro.csv',index=False)
    if promoted:
        dest=Path('artifacts/b6pro_long_best'); tmp=dest/'predictions.npz.tmp'
        np.savez_compressed(tmp,y=y.to_numpy(),oof=deliver_oof,test=deliver_te); tmp.replace(dest/'predictions.npz')
        sub.to_csv(dest/'submission_b6pro.csv',index=False); sub.to_csv('submissions/b6pro_closest/submission_b6pro.csv',index=False)
        (dest/'metrics.json').write_text(json.dumps({'experiment_id':'b6pro_long_best','spec':tag,'nested_oof_auc':deliver,'baseline_max3':B7_FLOOR,'gate_0_71':deliver>=GATE,'gap_to_0_71':GATE-deliver,'source':'b6pro_ultralong2'},indent=2))
    metrics={'best':tag,'nested':deliver,'promoted':promoted,'gate':deliver>=GATE,'closest_prev':CLOSEST,'top':sorted(results.items(),key=lambda kv:-kv[1])[:10]}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2), flush=True)
    print(f"GATE={'PASS' if deliver>=GATE else 'FAIL'} nested={deliver:.8f} promoted={promoted}", flush=True)
    return 0 if deliver>=GATE else 2

if __name__=='__main__':
    raise SystemExit(main())
