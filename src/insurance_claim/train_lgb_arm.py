"""LightGBM diversity arm on compressed numeric + low-card cats (NEW data)."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from insurance_claim.feature_blocks import DomainParseFeatureBlock, NumericPhysicsFeatureBlock, RawFeatureBlock
from insurance_claim.model import TARGET, audit_data, build_submission
from insurance_claim.train_lean_business import add_business_crosses, fit_edges

N_SPLITS=5
PARAMS=dict(n_estimators=3000, learning_rate=0.03, num_leaves=31, subsample=0.8, colsample_bytree=0.7,
            reg_lambda=5.0, reg_alpha=0.5, min_child_samples=40, objective='binary',
            n_jobs=-1, verbose=-1)

def build(Xtr,Xva,Xte):
    raw=RawFeatureBlock(drop_near_id_latent=True)
    parse=DomainParseFeatureBlock(); phys=NumericPhysicsFeatureBlock()
    tr=pd.concat([raw.fit_transform(Xtr), parse.fit_transform(Xtr), phys.fit_transform(
        pd.concat([Xtr.reset_index(drop=True), parse.fit_transform(Xtr)],axis=1).loc[:,lambda d:~d.columns.duplicated()]
    )],axis=1).loc[:,lambda d:~d.columns.duplicated()]
    # rebuild parse once
    parse=DomainParseFeatureBlock(); ptr=parse.fit_transform(Xtr); pva=parse.transform(Xva); pte=parse.transform(Xte)
    raw=RawFeatureBlock(drop_near_id_latent=True)
    rtr,rva,rte=raw.fit_transform(Xtr),raw.transform(Xva),raw.transform(Xte)
    def aug(a,b):
        return pd.concat([a.reset_index(drop=True),b.reset_index(drop=True)],axis=1).loc[:,lambda d:~d.columns.duplicated()]
    tr_aug,va_aug,te_aug=aug(Xtr,ptr),aug(Xva,pva),aug(Xte,pte)
    phys=NumericPhysicsFeatureBlock()
    ptr2,pva2,pte2=phys.fit_transform(tr_aug),phys.transform(va_aug),phys.transform(te_aug)
    tr=aug(aug(rtr,ptr),ptr2); va=aug(aug(rva,pva),pva2); te=aug(aug(rte,pte),pte2)
    edges=fit_edges(Xtr)
    keep=['days','condition','region','source','code','w1','w2','age_range','version']
    def with_raw(fe,rawdf):
        k=[c for c in keep if c in rawdf.columns]
        return pd.concat([fe.reset_index(drop=True), rawdf[k].reset_index(drop=True)],axis=1).loc[:,lambda d:~d.columns.duplicated()]
    tr=add_business_crosses(with_raw(tr,Xtr),edges)
    va=add_business_crosses(with_raw(va,Xva),edges).reindex(columns=tr.columns)
    te=add_business_crosses(with_raw(te,Xte),edges).reindex(columns=tr.columns)
    cat_cols=[c for c in tr.columns if (not pd.api.types.is_numeric_dtype(tr[c])) or str(c).startswith('biz_') or str(c) in
              {'region','source','version','code','t3','grades','month','t3_sfx','car_id','ver_era','car_token','t3_bin','t3_key'}]
    for c in cat_cols:
        for df in (tr,va,te):
            df[c]=df[c].astype('category') if c in df else None
    # numeric fill
    for c in tr.columns:
        if c in cat_cols: continue
        med=pd.to_numeric(tr[c],errors='coerce').median()
        for df in (tr,va,te):
            df[c]=pd.to_numeric(df[c],errors='coerce').fillna(med)
    return tr,va,te,cat_cols

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seeds',nargs='+',type=int,default=[2026])
    ap.add_argument('--output-dir',type=Path,default=Path('artifacts/lgb_arm')); args=ap.parse_args()
    train=pd.read_csv('train.csv'); test=pd.read_csv('test.csv'); sample=pd.read_csv('submit_sample.csv')
    audit_data(train,test,sample)
    y=train[TARGET].astype(int); feats=train.drop(columns=[TARGET])
    oofs=[]; tests=[]; folds=[]; t0=time.time()
    for seed in args.seeds:
        oof=np.zeros(len(train)); te=np.zeros(len(test))
        for fold,(a,b) in enumerate(StratifiedKFold(N_SPLITS,shuffle=True,random_state=seed).split(feats,y)):
            Xtr,Xva=feats.iloc[a].reset_index(drop=True),feats.iloc[b].reset_index(drop=True)
            ytr,yva=y.iloc[a].reset_index(drop=True),y.iloc[b].reset_index(drop=True)
            tr,va,te_fe,cats=build(Xtr,Xva,test.copy())
            model=LGBMClassifier(**PARAMS, random_state=seed+fold)
            model.fit(tr,ytr, eval_set=[(va,yva)], categorical_feature=cats,
                      callbacks=[early_stopping(120), log_evaluation(0)])
            oof[b]=model.predict_proba(va)[:,1]; te+=model.predict_proba(te_fe)[:,1]/N_SPLITS
            auc=float(roc_auc_score(yva,oof[b])); folds.append({'seed':seed,'fold':fold,'auc':auc})
            print(f'lgb seed={seed} fold={fold} auc={auc:.5f}', flush=True)
        print(f'lgb seed={seed} OOF={roc_auc_score(y,oof):.6f}', flush=True)
        oofs.append(oof); tests.append(te)
    oof=np.mean(np.vstack(oofs),0); te=np.mean(np.vstack(tests),0)
    metrics={'recipe':'lgb_arm','pooled_oof_auc':float(roc_auc_score(y,oof)),'folds':folds,
             'elapsed_sec':round(time.time()-t0,1),'gate_0_698':bool(roc_auc_score(y,oof)>=0.698)}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output_dir/'predictions.npz',oof=oof,test=te,y=y.to_numpy())
    build_submission(test,sample,te,args.output_dir/'submission_lgb.csv')
    (args.output_dir/'metrics.json').write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2))
if __name__=='__main__':
    main()
