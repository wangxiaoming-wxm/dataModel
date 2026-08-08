#!/usr/bin/env python3
"""XGBoost full-data resid-matrix arm for diversity vs CatBoost B7."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from insurance_claim.b6pro_fusion import nested_select_rule, apply_rule
from insurance_claim.b6pro_long_resid import build_long_resid_matrix

B7_FLOOR=0.7027049552615718; GATE=0.71
CLOSEST=float(json.load(open('artifacts/b6pro_long_best/metrics.json'))['nested_oof_auc'])

def main():
    train=pd.read_csv('train.csv'); test=pd.read_csv('test.csv'); sample=pd.read_csv('submit_sample.csv')
    y=train['label'].astype(int); features=train.drop(columns=['label'])
    days=features['days'].to_numpy(float); long=days>=3000
    b7=np.load('reference/b7_closest/predictions.npz'); fr=np.load('artifacts/b6pro_frozen/predictions.npz')
    max3=np.maximum.reduce([b7['gap'],b7['gap_bag'],b7['plus']])
    tmax=np.maximum.reduce([fr['test_gap'],fr['test_gap_bag'],fr['test_plus']])
    cur=np.load('artifacts/b6pro_long_best/predictions.npz')
    fk=np.load('artifacts/b6pro_full_keepx/predictions.npz')
    seeds=[2026,2027,2028,2029]
    oof_acc=np.zeros(len(y)); te_acc=np.zeros(len(test))
    params=dict(n_estimators=3500, learning_rate=0.02, max_depth=6, min_child_weight=40,
                subsample=0.85, colsample_bytree=0.7, reg_lambda=8.0, reg_alpha=0.5,
                objective='binary:logistic', eval_metric='auc', tree_method='hist',
                early_stopping_rounds=150, n_jobs=4)
    for seed in seeds:
        oof=np.zeros(len(y)); pte=np.zeros(len(test))
        for fold,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(features,y)):
            trd,vad,ted,cats=build_long_resid_matrix(features.iloc[tr].reset_index(drop=True), y.iloc[tr].to_numpy(),
                                                     features.iloc[va].reset_index(drop=True), test.copy())
            model=xgb.XGBClassifier(**{**params, 'random_state':seed+fold})
            model.fit(trd, y.iloc[tr], eval_set=[(vad, y.iloc[va])], verbose=False)
            oof[va]=model.predict_proba(vad)[:,1]
            pte+=model.predict_proba(ted)[:,1]/5
            print(f'xgb s{seed} f{fold} {roc_auc_score(y.iloc[va],oof[va]):.5f} best={model.best_iteration}', flush=True)
        print(f'xgb s{seed} OOF={roc_auc_score(y,oof):.6f} long={roc_auc_score(y.to_numpy()[long],oof[long]):.6f}', flush=True)
        oof_acc+=oof; te_acc+=pte
    oof_x, te_x = oof_acc/len(seeds), te_acc/len(seeds)
    print('pooled', roc_auc_score(y,oof_x), 'corr(max3)', np.corrcoef(oof_x,max3)[0,1], flush=True)
    arms={
        'raw':(oof_x,te_x),
        'max_m3':(np.maximum(max3,oof_x),np.maximum(tmax,te_x)),
        'mean_m3':(0.5*(max3+oof_x),0.5*(tmax+te_x)),
        'mean_kx':(0.5*(fk['oof_k']+oof_x),0.5*(fk['te_k']+te_x)),
        'max_kx':(np.maximum(fk['oof_k'],oof_x),np.maximum(fk['te_k'],te_x)),
    }
    results={}; best_name=None; best_res=None; best_pair=None
    for name,(oa,ta) in arms.items():
        for tag,oof_arms,te_arms in [
            (f'b7+{name}',[b7['gap'],b7['gap_bag'],b7['plus'],oa],[fr['test_gap'],fr['test_gap_bag'],fr['test_plus'],ta]),
            (f'cur+{name}',[b7['gap'],b7['gap_bag'],b7['plus'],cur['oof'],oa],[fr['test_gap'],fr['test_gap_bag'],fr['test_plus'],cur['test'],ta]),
            (f'b7+kx+{name}',[b7['gap'],b7['gap_bag'],b7['plus'],fk['oof_k'],oa],[fr['test_gap'],fr['test_gap_bag'],fr['test_plus'],fk['te_k'],ta]),
            (f'max3×{name}',[max3,oa],[tmax,ta]),
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
    out=Path('artifacts/b6pro_xgb_full'); out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'predictions.npz', y=y.to_numpy(), oof=deliver_oof, test=deliver_test, oof_x=oof_x, te_x=te_x)
    lab=[c for c in sample.columns if c!='id'][0]
    sub=sample.copy(); sub[lab]=deliver_test; sub.to_csv(out/'submission_b6pro.csv',index=False)
    if promoted:
        dest=Path('artifacts/b6pro_long_best')
        np.savez_compressed(dest/'predictions.npz', y=y.to_numpy(), oof=deliver_oof, test=deliver_test, arm=oof_x)
        sub.to_csv(dest/'submission_b6pro.csv',index=False)
        Path('submissions/b6pro_closest').mkdir(parents=True,exist_ok=True)
        sub.to_csv('submissions/b6pro_closest/submission_b6pro.csv',index=False)
        (dest/'metrics.json').write_text(json.dumps({'experiment_id':'b6pro_long_best','spec':best_name,'nested_oof_auc':deliver,'baseline_max3':B7_FLOOR,'gate_0_71':deliver>=GATE,'gap_to_0_71':GATE-deliver,'source':'b6pro_xgb_full'},indent=2))
    metrics={'best':best_name,'nested':deliver,'xgb':float(roc_auc_score(y,oof_x)),'corr':float(np.corrcoef(oof_x,max3)[0,1]),'promoted':promoted,'gate':deliver>=GATE,'top':sorted(results.items(),key=lambda kv:-kv[1])[:12]}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2), flush=True)
    print(f"GATE={'PASS' if deliver>=GATE else 'FAIL'} nested={deliver:.8f} promoted={promoted}", flush=True)
    return 0 if deliver>=GATE else 2
if __name__=='__main__':
    raise SystemExit(main())
