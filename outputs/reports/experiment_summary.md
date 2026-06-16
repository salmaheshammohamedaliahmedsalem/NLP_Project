# Experiment Summary

## Table 1: SemEval Results

| model | train_dataset | test_dataset | accuracy | precision | recall | f1 | roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M1_Word_TFIDF_LogReg | semeval | semeval | 0.9021 | 0.9003 | 0.8918 | 0.8960 | 0.9665 |
| M2_Char_TFIDF_LinearSVM | semeval | semeval | 0.9430 | 0.9418 | 0.9375 | 0.9396 | 0.9879 |
| M3_Stylometric_RandomForest | semeval | semeval | 0.8414 | 0.8408 | 0.8199 | 0.8302 | 0.9191 |
| M4_Hybrid_TFIDF_Stylometric | semeval | semeval | 0.9475 | 0.9484 | 0.9402 | 0.9443 | 0.9893 |

## Table 2: RAID Results

| model | train_dataset | test_dataset | accuracy | precision | recall | f1 | roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M1_Word_TFIDF_LogReg | raid | raid | 0.8699 | 0.9143 | 0.8159 | 0.8623 | 0.9517 |
| M2_Char_TFIDF_LinearSVM | raid | raid | 0.9533 | 0.9624 | 0.9433 | 0.9528 | 0.9905 |
| M3_Stylometric_RandomForest | raid | raid | 0.8769 | 0.9346 | 0.8102 | 0.8680 | 0.9555 |
| M4_Hybrid_TFIDF_Stylometric | raid | raid | 0.9491 | 0.9789 | 0.9178 | 0.9474 | 0.9890 |

## Table 3: Old Dataset Results

| model | train_dataset | test_dataset | accuracy | precision | recall | f1 | roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M1_Word_TFIDF_LogReg | old_ai | old_ai | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| M2_Char_TFIDF_LinearSVM | old_ai | old_ai | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| M3_Stylometric_RandomForest | old_ai | old_ai | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| M4_Hybrid_TFIDF_Stylometric | old_ai | old_ai | 0.9994 | 0.9988 | 1.0000 | 0.9994 | 1.0000 |

## Table 4: Cross-Dataset Results

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

## Table 5: Old Dataset Cross-Evaluation

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

## Table 6: Arabic POC Results

| model | train_dataset | test_dataset | accuracy | precision | recall | f1 | roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M2_Char_TFIDF_LinearSVM | arabic | arabic | 0.9885 | 0.9977 | 0.9897 | 0.9937 | 0.9980 |
| M4_Hybrid_TFIDF_Stylometric | arabic | arabic | 0.9889 | 0.9954 | 0.9925 | 0.9940 | 0.9978 |
