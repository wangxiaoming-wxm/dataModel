#!/usr/bin/env python3
"""Full-data keepx + aging arms fused with B7 / closest."""
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

B7_FLOOR=0.7027049552615718; GATE=0.71; CLOSEST=0.7054764270400189
PARAMS={**PARAMS_GAP_BAG,'thread_count':4,'iterations':3000,'od_wait':150}

def main():
    train=pd.read_csv('train.csv'); test=pd.read_csv('test.csv'); sample=pd.read_csv('submit_sample.csv')
    y=train['label'].astype(int); features=train.drop(columns=['label'])
    days=features['days'].to_numpy(float); long=days>=3000
    b7=np.load('reference/b7_closest/predictions.npz'); fr=np.load('artifacts/b6pro_frozen/predictions.npz')
    max3=np.maximum.reduce([b7['gap'],b7['gap_bag'],b7['plus']])
    tmax=np.maximum.reduce([fr['test_gap'],fr['test_gap_bag'],fr['test_plus']])
    cur=np.load('artifacts/b6pro_long_best/predictions.npz')
    def run(builder, seeds, tag):
        oof_acc=np.zeros(len(y)); te_acc=np.zeros(len(test))
        for seed in seeds:
            oof=np.zeros(len(y)); pte=np.zeros(len(test))
            for fold,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(features,y)):
                trd,vad,ted,cats=builder(features.iloc[tr].reset_index(drop=True), features.iloc[va].reset_index(drop=True), test.copy())
                model=CatBoostClassifier(**{**PARAMS,'random_seed':seed+fold})
                model.fit(trd,y.iloc[tr],eval_set=(vad,y.iloc[va]),cat_features=cats,use_best_model=True)
                oof[va]=model.predict_proba(vad)[:,1]
                pte+=model.predict_proba(ted)[:,1]/5
                print(f'{tag} s{seed} f{fold} {roc_auc_score(y.iloc[va],oof[va]):.5f}', flush=True)
            print(f'{tag} s{seed} OOF={roc_auc_score(y,oof):.6f} long={roc_auc_score(y.to_numpy()[long],oof[long]):.6f}', flush=True)
            oof_acc+=oof; te_acc+=pte
        return oof_acc/len(seeds), te_acc/len(seeds)
    seeds=[2026,2027,2028,2029]
    print('=== FULL keepx ===', flush=True)
    oof_k, te_k = run(build_long_keepx, seeds, 'full_kx')
    print('pooled', roc_auc_score(y,oof_k), 'long', roc_auc_score(y.to_numpy()[long],oof_k[long]), 'corr', np.corrcoef(oof_k,max3)[0,1], flush=True)
    print('=== FULL aging ===', flush=True)
    oof_a, te_a = run(build_long_aging, seeds, 'full_ag')
    print('pooled', roc_auc_score(y,oof_a), 'long', roc_auc_score(y.to_numpy()[long],oof_a[long]), 'corr', np.corrcoef(oof_a,max3)[0,1], flush=True)
    nodays=np.load('artifacts/b6pro_nodays_keepx/predictions.npz')
    oof_nd, te_nd = nodays['oof_nd'], nodays['test_nd']
    mean_ka=0.5*(oof_k+oof_a); tmean=0.5*(te_k+te_a)
    arms={
        'kx':(oof_k,te_k),'ag':(oof_a,te_a),'mean_ka':(mean_ka,tmean),
        'max_ka':(np.maximum(oof_k,oof_a),np.maximum(te_k,te_a)),
        'max_m3_kx':(np.maximum(max3,oof_k),np.maximum(tmax,te_k)),
        'mean_m3_kx':(0.5*(max3+oof_k),0.5*(tmax+te_k)),
        'max3arms':(np.maximum.reduce([max3,oof_k,oof_a]),np.maximum.reduce([tmax,te_k,te_a])),
        'mean_kx_nd':(0.5*(oof_k+oof_nd),0.5*(te_k+te_nd)),
    }
    results={}; best_name=None; best_res=None; best_pair=None
    for name,(oof_arm,te_arm) in arms.items():
        for tag,oof_arms,te_arms in [
            (f'b7+{name}',[b7['gap'],b7['gap_bag'],b7['plus'],oof_arm],[fr['test_gap'],fr['test_gap_bag'],fr['test_plus'],te_arm]),
            (f'cur+{name}',[b7['gap'],b7['gap_bag'],b7['plus'],cur['oof'],oof_arm],[fr['test_gap'],fr['test_gap_bag'],fr['test_plus'],cur['test'],te_arm]),
            (f'max3×{name}',[max3,oof_arm],[tmax,te_arm]),
        ]:
            res=nested_select_rule(y.to_numpy(), oof_arms)
            results[tag]=float(res['nested_oof_auc'])
            print(f'{tag}: {res["nested_oof_auc"]:.8f}', flush=True)
            if best_res is None or res['nested_oof_auc']>best_res['nested_oof_auc']:
                best_name,best_res,best_pair=tag,res,(oof_arms,te_arms)
    for tag,oof_arms,te_arms in [
        ('b7+kx+ag',[b7['gap'],b7['gap_bag'],b7['plus'],oof_k,oof_a],[fr['test_gap'],fr['test_gap_bag'],fr['test_plus'],te_k,te_a]),
        ('b7+kx+ag+nd',[b7['gap'],b7['gap_bag'],b7['plus'],oof_k,oof_a,oof_nd],[fr['test_gap'],fr['test_gap_bag'],fr['test_plus'],te_k,te_a,te_nd]),
        ('b7+kx+cur',[b7['gap'],b7['gap_bag'],b7['plus'],oof_k,cur['oof']],[fr['test_gap'],fr['test_gap_bag'],fr['test_plus'],te_k,cur['test']]),
    ]:
        res=nested_select_rule(y.to_numpy(), oof_arms)
        results[tag]=float(res['nested_oof_auc'])
        print(f'{tag}: {res["nested_oof_auc"]:.8f}', flush=True)
        if best_res is None or res['nested_oof_auc']>best_res['nested_oof_auc']:
            best_name,best_res,best_pair=tag,res,(oof_arms,te_arms)
    deliver=best_res['nested_oof_auc']; deliver_oof=best_res['nested_oof']
    deliver_test=apply_rule(best_res['selected_rule'], best_pair[1])
    if deliver<B7_FLOOR:
        deliver=float(roc_auc_score(y,max3)); deliver_oof,deliver_test=max3,tmax; best_name='b7_fallback'
    promoted=deliver>CLOSEST+1e-12
    out=Path('artifacts/b6pro_full_keepx'); out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'predictions.npz', y=y.to_numpy(), oof=deliver_oof, test=deliver_test, oof_k=oof_k, te_k=te_k, oof_a=oof_a, te_a=te_a)
    lab=[c for c in sample.columns if c!='id'][0]
    sub=sample.copy(); sub[lab]=deliver_test; sub.to_csv(out/'submission_b6pro.csv',index=False)
    if promoted:
        dest=Path('artifacts/b6pro_long_best')
        np.savez_compressed(dest/'predictions.npz', y=y.to_numpy(), oof=deliver_oof, test=deliver_test, arm=oof_k)
        sub.to_csv(dest/'submission_b6pro.csv',index=False)
        Path('submissions/b6pro_closest').mkdir(parents=True,exist_ok=True)
        sub.to_csv('submissions/b6pro_closest/submission_b6pro.csv',index=False)
        (dest/'metrics.json').write_text(json.dumps({'experiment_id':'b6pro_long_best','spec':best_name,'nested_oof_auc':deliver,'baseline_max3':B7_FLOOR,'gate_0_71':deliver>=GATE,'gap_to_0_71':GATE-deliver,'source':'b6pro_full_keepx'},indent=2))
    metrics={'best':best_name,'nested':deliver,'kx':float(roc_auc_score(y,oof_k)),'ag':float(roc_auc_score(y,oof_a)),'promoted':promoted,'gate':deliver>=GATE,'top':sorted(results.items(),key=lambda kv:-kv[1])[:15]}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2), flush=True)
    print(f"GATE={'PASS' if deliver>=GATE else 'FAIL'} nested={deliver:.8f} promoted={promoted}", flush=True)
    return 0 if deliver>=GATE else 2
if __name__=='__main__':
    raise SystemExit(main())
