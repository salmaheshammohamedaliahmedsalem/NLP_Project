# Table 4: Cross-Dataset Results

| model | train_dataset | test_dataset | accuracy | precision | recall | f1 | roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M1_Word_TFIDF_LogReg | semeval | raid | 0.7992 | 0.8459 | 0.7309 | 0.7842 | 0.8455 |
| M2_Char_TFIDF_LinearSVM | semeval | raid | 0.8091 | 0.8811 | 0.7139 | 0.7887 | 0.8374 |
| M3_Stylometric_RandomForest | semeval | raid | 0.7298 | 0.6976 | 0.8102 | 0.7497 | 0.7775 |
| M4_Hybrid_TFIDF_Stylometric | semeval | raid | 0.8034 | 0.8821 | 0.6997 | 0.7804 | 0.8340 |
| M1_Word_TFIDF_LogReg | raid | semeval | 0.5721 | 0.5381 | 0.6722 | 0.5977 | 0.6220 |
| M2_Char_TFIDF_LinearSVM | raid | semeval | 0.5935 | 0.6150 | 0.3755 | 0.4663 | 0.6407 |
| M3_Stylometric_RandomForest | raid | semeval | 0.5241 | 0.4973 | 0.5746 | 0.5332 | 0.5359 |
| M4_Hybrid_TFIDF_Stylometric | raid | semeval | 0.4944 | 0.4691 | 0.5250 | 0.4955 | 0.4549 |
