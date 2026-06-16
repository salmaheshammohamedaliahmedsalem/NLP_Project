# Table 2: RAID Results

| model | train_dataset | test_dataset | accuracy | precision | recall | f1 | roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M1_Word_TFIDF_LogReg | raid | raid | 0.8699 | 0.9143 | 0.8159 | 0.8623 | 0.9517 |
| M2_Char_TFIDF_LinearSVM | raid | raid | 0.9533 | 0.9624 | 0.9433 | 0.9528 | 0.9905 |
| M3_Stylometric_RandomForest | raid | raid | 0.8769 | 0.9346 | 0.8102 | 0.8680 | 0.9555 |
| M4_Hybrid_TFIDF_Stylometric | raid | raid | 0.9491 | 0.9789 | 0.9178 | 0.9474 | 0.9890 |
