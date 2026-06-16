# Table 5: Old Dataset Cross-Evaluation

| model | train_dataset | test_dataset | accuracy | precision | recall | f1 | roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M1_Word_TFIDF_LogReg | semeval | old_ai | 0.5058 | 0.5058 | 1.0000 | 0.6718 | 0.5017 |
| M2_Char_TFIDF_LinearSVM | semeval | old_ai | 0.5045 | 0.5052 | 0.9964 | 0.6704 | 0.4008 |
| M3_Stylometric_RandomForest | semeval | old_ai | 0.4876 | 0.4961 | 0.8297 | 0.6209 | 0.3772 |
| M4_Hybrid_TFIDF_Stylometric | semeval | old_ai | 0.5070 | 0.5064 | 0.9976 | 0.6718 | 0.3626 |
| M1_Word_TFIDF_LogReg | raid | old_ai | 0.5143 | 0.5101 | 0.9976 | 0.6751 | 0.5417 |
| M2_Char_TFIDF_LinearSVM | raid | old_ai | 0.4991 | 0.5025 | 0.9688 | 0.6618 | 0.4550 |
| M3_Stylometric_RandomForest | raid | old_ai | 0.5058 | 0.5058 | 1.0000 | 0.6718 | 0.3774 |
| M4_Hybrid_TFIDF_Stylometric | raid | old_ai | 0.5185 | 0.5365 | 0.3525 | 0.4255 | 0.5184 |
| M1_Word_TFIDF_LogReg | old_ai | semeval | 0.5288 | 0.5385 | 0.0250 | 0.0477 | 0.4462 |
| M1_Word_TFIDF_LogReg | old_ai | raid | 0.4965 | 0.4595 | 0.0482 | 0.0872 | 0.3628 |
| M2_Char_TFIDF_LinearSVM | old_ai | semeval | 0.5301 | 0.6022 | 0.0188 | 0.0364 | 0.3908 |
| M2_Char_TFIDF_LinearSVM | old_ai | raid | 0.4908 | 0.3600 | 0.0255 | 0.0476 | 0.3900 |
| M3_Stylometric_RandomForest | old_ai | semeval | 0.4601 | 0.4640 | 0.9135 | 0.6154 | 0.4676 |
| M3_Stylometric_RandomForest | old_ai | raid | 0.4187 | 0.4391 | 0.5921 | 0.5042 | 0.3580 |
| M4_Hybrid_TFIDF_Stylometric | old_ai | semeval | 0.5410 | 0.6602 | 0.0605 | 0.1109 | 0.6178 |
| M4_Hybrid_TFIDF_Stylometric | old_ai | raid | 0.5276 | 0.8065 | 0.0708 | 0.1302 | 0.5016 |
