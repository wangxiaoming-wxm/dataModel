# Business Feature Synergy Audit (numbers)

- n=14930, claim_rate=0.100201, positives=1496
- days tercile OR: low=0.570, mid=1.068, high=1.397
- condition lowest/highest decile OR: 1.527 / 0.624
- livability R² on region: 0.9896
- corr(cc,V)=0.939, corr(x19,V)=0.997

## Cross TE leakage gaps (leaky − OOF)

- **region×days5×version**: leaky=0.7574, oof=0.5593, gap=0.1981, cells=1410, n<20=1265 (row_share=0.354), CramerV=0.3184
- **car×version×days5**: leaky=0.7139, oof=0.5532, gap=0.1608, cells=887, n<20=759 (row_share=0.271), CramerV=0.2707
- **t3×code**: leaky=0.6241, oof=0.5230, gap=0.1011, cells=256, n<20=89 (row_share=0.066), CramerV=0.1447
- **car×version**: leaky=0.5924, oof=0.5112, gap=0.0812, cells=203, n<20=116 (row_share=0.065), CramerV=0.1191
- **source×version**: leaky=0.5924, oof=0.5112, gap=0.0812, cells=203, n<20=116 (row_share=0.065), CramerV=0.1191
- **region×car**: leaky=0.6134, oof=0.5490, gap=0.0644, cells=211, n<20=98 (row_share=0.047), CramerV=0.1329
- **region×days_bin10**: leaky=0.6592, oof=0.5974, gap=0.0619, cells=200, n<20=64 (row_share=0.053), CramerV=0.1707
- **days_bin5×version**: leaky=0.6210, oof=0.5749, gap=0.0461, cells=95, n<20=0 (row_share=0.000), CramerV=0.1307
- **region×days_bin5**: leaky=0.6376, oof=0.6001, gap=0.0375, cells=100, n<20=9 (row_share=0.009), CramerV=0.1468
- **source×days_bin5**: leaky=0.6269, oof=0.6024, gap=0.0245, cells=55, n<20=4 (row_share=0.004), CramerV=0.1366
- **t3_sfx×code×days_bin5**: leaky=0.6194, oof=0.6018, gap=0.0176, cells=35, n<20=0 (row_share=0.000), CramerV=0.1275
- **t3_sfx×code**: leaky=0.5257, oof=0.5113, gap=0.0144, cells=7, n<20=0 (row_share=0.000), CramerV=0.0287
- **age_coarse×days_bin5**: leaky=0.6080, oof=0.5950, gap=0.0130, cells=25, n<20=2 (row_share=0.002), CramerV=0.1145
- **days_bin5×condition_bin5**: leaky=0.6247, oof=0.6129, gap=0.0118, cells=25, n<20=0 (row_share=0.000), CramerV=0.1312
- **w_pair×days_bin5**: leaky=0.6074, oof=0.5958, gap=0.0116, cells=20, n<20=0 (row_share=0.000), CramerV=0.1147
