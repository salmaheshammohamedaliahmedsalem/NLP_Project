# Antigravity Instructions: Fix the AI-vs-Human Detection Project Code

## Main problem
The notebook currently reports **100% accuracy / F1 / ROC-AUC for every model**, including simple TF-IDF Logistic Regression, SVM, stylometric-only, transformer, and hybrid models. This is a major red flag. Do **not** focus only on making the code run. The goal is to make the evaluation scientifically valid and remove leakage / over-easy evaluation.

## What to fix immediately

### 1. Add a leakage audit before training
Create a new section after data loading/cleaning called:

```python
# Section: Leakage and Dataset Audit
```

It must check and print:

- number of rows before and after cleaning
- label distribution
- domain/type distribution by label
- source distribution by label
- ai_model distribution by label
- word_count and char_count mean/std/min/max by label
- duplicate count based on exact `content`
- duplicate count based on normalized content
- near-duplicate similarity warning

Important: check whether columns like `source`, `ai_model`, `type`, `language`, `word_count`, `char_count`, `complexity_score`, or `is_multiline_code` are strongly correlated with the label.

Example red flags to detect:

```python
pd.crosstab(df['source'], df['label'], normalize='index')
pd.crosstab(df['ai_model'].fillna('human_or_missing'), df['label'], normalize='index')
pd.crosstab(df['type'], df['label'], normalize='index')
```

If `ai_model` is missing mostly for human and present mostly for AI, report it as leakage and never use it directly or indirectly.

---

### 2. Replace random train/test split with stricter split strategies
The current random split is too easy. Implement at least these three evaluations:

#### A. Random stratified split
Keep this only as a baseline.

#### B. Prompt-grouped split
Use `GroupShuffleSplit` or `GroupKFold` where the group is the `prompt` column.

Reason: if the same prompt appears in both train and test, the model may memorize prompt/topic/template patterns.

Use:

```python
from sklearn.model_selection import GroupShuffleSplit

groups = df['prompt'].astype(str)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(df, df['label_num'], groups=groups))
```

#### C. Domain holdout split
Train on one domain and test on the other, for example:

- train on `text`, test on `code`
- train on `code`, test on `text`

This is essential because the project claims domain-aware generalization.

---

### 3. Remove metadata leakage from modeling
The model should not use any feature that gives away the label.

Do **not** use these columns as model inputs:

```python
id
prompt
source
ai_model
label
word_count
char_count
complexity_score
is_multiline_code
language
```

Only use:

- `content` as text input
- safe domain label only if it is balanced and not label-leaking
- stylometric features computed from the actual content only

If using domain embeddings, first prove that `type/domain` is not just a proxy for label. If it is correlated with label, remove domain embedding from the main model and keep it only as an ablation experiment.

---

### 4. Add near-duplicate removal
Exact duplicate removal is not enough. Add normalized duplicate and near-duplicate detection.

Create a normalized text column:

```python
def normalize_for_dedup(text):
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z0-9 ]+', '', text)
    return text.strip()

df['content_norm'] = df[TEXT_COL].apply(normalize_for_dedup)
df = df.drop_duplicates(subset=['content_norm']).reset_index(drop=True)
```

Then add a near-duplicate check using TF-IDF cosine similarity on a sample or using MinHash/SimHash if available. At minimum, report suspicious pairs with cosine similarity above 0.90.

---

### 5. Add an external/challenge test set
The current dataset is too easy. Create a new challenge test set that is **not used in training**.

Add examples from:

- unseen prompts
- unseen topics
- different AI models if possible
- human-written messy text
- AI text rewritten to sound human
- short texts
- long texts
- code and non-code separately

Save it as:

```text
challenge_test.csv
```

Columns:

```text
content,label,type,source_note
```

Then evaluate every model on this file separately.

---

### 6. Add sanity checks
Add these sanity tests before trusting results:

#### Label shuffle test
Randomly shuffle labels and train a simple TF-IDF model. Accuracy should drop near 50%.

```python
shuffled_y = np.random.permutation(y_train)
```

If shuffled-label accuracy is still high, there is leakage.

#### Text-only check
Train using only `content`. No metadata.

#### Length-only check
Train using only text length features. If this gives very high accuracy, the dataset is biased by length.

#### Prompt-only check
Train using `prompt` only. If this gives high accuracy, the prompt is leaking label/topic/template information.

---

### 7. Fix the conclusion text
Do not claim the hybrid model is better if all models score 1.0.

Replace claims like:

> hybrid model outperforms all baselines

with:

> all models achieved perfect scores on the random split, which indicates that the current dataset/split is likely too easy or affected by leakage. Therefore, stricter grouped and domain-holdout evaluations are required before making any strong generalization claims.

The final report must honestly say that 100% results are suspicious, not impressive.

---

### 8. Add comparison table for all split types
Create one final table with rows like:

```text
Model | Random Split F1 | Prompt-Grouped F1 | Domain Holdout F1 | Challenge Test F1
```

This is more valuable than only reporting random test accuracy.

---

### 9. Keep the models, but restructure the notebook
Keep the current models, but restructure the code in this order:

1. imports and config
2. load data
3. clean data
4. leakage audit
5. safe feature creation
6. split strategies
7. baseline models
8. transformer model
9. hybrid model
10. sanity checks
11. challenge evaluation
12. final honest comparison and conclusion

---

## Final required output files
The fixed project must generate:

```text
results/leakage_audit.csv
results/random_split_results.csv
results/prompt_grouped_results.csv
results/domain_holdout_results.csv
results/challenge_test_results.csv
results/final_comparison_all_splits.csv
figures/final_comparison_all_splits.png
```

## Most important rule
Do not optimize the code to keep the 100% result. Optimize it to find out whether the 100% result is real. If stricter testing drops performance, that is good and scientifically honest.
