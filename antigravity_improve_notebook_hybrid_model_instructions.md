# Instructions for Antigravity: Improve the Hybrid Model in the Notebook

Please edit only this notebook:

`01_ai_vs_human_detection_project.ipynb`

Do not run heavy training locally. I will run the notebook myself on Google Colab.

## Main objective

The current hybrid model inside the notebook has potential, but it is not working correctly. It collapses toward one class and does not outperform the baselines. Please modify the notebook hybrid model so it follows the successful approach used in:

`improved_distilbert_stylometric_vs_code.py`

The final notebook should still compare multiple approaches, but the final proposed model should become:

```text
Improved Hybrid Model = DistilBERT + Stylometric Features
```

The goal is to make the notebook version of the hybrid model more stable, balanced, and suitable for the final paper.

---

## Important context

The old results were suspicious because the original dataset had leakage. We now use the cleaned dataset:

`ai_vs_human_content_v2_FIXED_clean_no_leakage.csv`

The notebook must use this cleaned dataset by default.

The cleaned dataset columns are expected to be:

```python
content
label
type
topic
language
word_count
char_count
complexity_score
is_multiline_code
```

Do not use leaking columns such as:

```python
source
ai_model
id
prompt
```

If these columns exist, they must not be used as features.

---

## Required changes

### 1. Keep the notebook comparison structure

Do not delete the baseline comparison section.

The notebook should still compare:

1. TF-IDF + Logistic Regression
2. TF-IDF + SVM
3. Stylometric-only model
4. Transformer baseline
5. Initial hybrid model, if kept
6. Final improved hybrid model

The final model should be clearly named:

```text
Improved DistilBERT + Stylometric Hybrid
```

---

### 2. Remove or disable domain embedding from the hybrid model

The current notebook hybrid model may use domain/type embeddings.

Please remove this from the final hybrid model.

Reason:

The cleaned dataset is mostly or entirely text-only, so domain/type does not provide useful information. It can also create instability or shortcut learning.

Add a config flag:

```python
USE_DOMAIN_FEATURE = False
```

Default must be:

```python
USE_DOMAIN_FEATURE = False
```

The final improved hybrid should only use:

```text
DistilBERT text embedding + stylometric features
```

---

### 3. Use cleaned content only

All models must use:

```python
df["content"]
```

from the fixed cleaned CSV.

If the notebook still creates a column like:

```python
content_clean
clean_content
text_clean
```

that is okay, but make sure the text no longer contains:

```text
AI-generated
(AI-generated)
# AI-generated
```

Add this check:

```python
marker_count = df["content"].astype(str).str.contains("AI-generated", case=False, na=False).sum()
print("AI-generated marker count:", marker_count)
```

If `marker_count > 0`, remove the markers before training.

---

### 4. Add robust data checks before training

Before splitting, print:

```python
print("Dataset shape:", df.shape)
print("Label counts:")
print(df["label"].value_counts())
print("Duplicate content rows:", df.duplicated("content", keep=False).sum())
print("Unique content:", df["content"].nunique())
```

Also check if any cleaned text appears with both labels:

```python
label_counts_per_text = df.groupby("content")["label"].nunique()
conflicting = label_counts_per_text[label_counts_per_text > 1]
print("Texts with conflicting labels:", len(conflicting))
```

If conflicts exist, remove them:

```python
df = df[~df["content"].isin(conflicting.index)].copy()
```

Then drop duplicates:

```python
df = df.drop_duplicates(subset=["content"]).reset_index(drop=True)
```

---

### 5. Use train / validation / test split

The final improved hybrid must not train and directly evaluate on test.

Use:

```text
train = 64%
validation = 16%
test = 20%
```

or similar.

Use stratified splitting:

```python
train_test_split(..., stratify=y, random_state=42)
```

Print:

```python
Train size
Validation size
Test size
Train label distribution
Validation label distribution
Test label distribution
```

---

### 6. Extract stylometric features properly

Add or improve a stylometric feature function similar to this:

```python
def stylometric_features(text):
    ...
```

It should include:

```python
char_count
word_count
sentence_count
avg_word_len
avg_sentence_len
type_token_ratio
repeated_word_ratio
punct_ratio
digit_ratio
uppercase_ratio
newline_count
code_symbol_ratio
comma_count
semicolon_count
bracket_count
quote_count
```

Scale these features using:

```python
StandardScaler()
```

Fit the scaler only on training features:

```python
scaler.fit(X_style_train)
```

Then transform validation and test.

Do not fit the scaler on the full dataset.

---

### 7. Use DistilBERT text encoder with mean pooling

The improved hybrid should use:

```python
distilbert-base-uncased
```

or keep the existing transformer if already loaded, but the architecture must use mean pooling.

Use mean pooling over token embeddings, not only the first token.

Example:

```python
def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts
```

---

### 8. Build the final improved hybrid architecture

The model should be:

```text
Input text
→ DistilBERT
→ mean pooling
→ text embedding

Input text
→ stylometric feature extraction
→ StandardScaler
→ dense projection

Text embedding + stylometric projection
→ concatenation
→ dropout
→ dense layer
→ binary output logit
```

Suggested model:

```python
class ImprovedDistilBertStylometricClassifier(nn.Module):
    def __init__(self, model_name, num_style_features, dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        self.style_projection = nn.Sequential(
            nn.Linear(num_style_features, 32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size + 32, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask, style_features):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_embedding = mean_pooling(outputs.last_hidden_state, attention_mask)
        style_embedding = self.style_projection(style_features)
        combined = torch.cat((text_embedding, style_embedding), dim=1)
        logits = self.classifier(combined).squeeze(1)
        return logits
```

---

### 9. Use BCEWithLogitsLoss, AdamW, dropout, and gradient clipping

Use:

```python
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
dropout = 0.3
```

During training:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

### 10. Tune classification threshold on validation set

Do not always use threshold 0.5 without checking.

Add a function:

```python
def find_best_threshold(y_true, y_prob):
    best_threshold = 0.5
    best_score = -1

    for threshold in np.arange(0.10, 0.91, 0.01):
        preds = (y_prob >= threshold).astype(int)
        score = f1_score(y_true, preds, average="macro")

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return float(best_threshold), float(best_score)
```

Use the best validation threshold for final test evaluation.

---

### 11. Save the best model based on validation Macro-F1

Do not automatically use the final epoch.

At each epoch:

1. Train one epoch.
2. Predict validation probabilities.
3. Find best validation threshold.
4. Calculate validation Macro-F1.
5. Save model if validation Macro-F1 improves.

Example:

```python
if val_macro_f1 > best_val_macro_f1:
    best_val_macro_f1 = val_macro_f1
    best_threshold = threshold
    torch.save(model.state_dict(), best_path)
```

Then before final testing:

```python
model.load_state_dict(torch.load(best_path, map_location=device))
```

---

### 12. Add prediction distribution checks

After validation and test predictions, print:

```python
print("Prediction distribution:")
print(pd.Series(y_pred).value_counts())
```

Also print:

```python
print("True label distribution:")
print(pd.Series(y_true).value_counts())
```

This is important because the earlier hybrid model collapsed toward one class.

If one class has zero predictions, print a warning:

```python
WARNING: Model predicted only one class.
```

---

### 13. Report balanced metrics

For the final improved hybrid, report:

```python
Accuracy
Balanced Accuracy
Macro-F1
AI F1
Human F1
ROC-AUC
Classification report
Confusion matrix
```

Use:

```python
classification_report(y_test_true, y_test_pred, target_names=["Human", "AI"])
confusion_matrix(y_test_true, y_test_pred)
```

---

### 14. Save stylometric signature table

After training, add a stylometric analysis table comparing AI and human averages.

Save it as:

```python
improved_stylometric_signature_table.csv
```

Add columns:

```python
ai
human
difference_ai_minus_human
```

Sort by absolute or signed difference.

---

### 15. Update final comparison table

At the end of the notebook, include a final table comparing all approaches.

Add the improved hybrid result as the final row:

```text
Improved DistilBERT + Stylometric Hybrid
```

This should be treated as the proposed final model.

The table should include:

```python
Model
Accuracy
Balanced Accuracy
Precision
Recall
Macro-F1
ROC-AUC
Main observation
```

For the failed initial hybrid, do not hide the failure. Mention:

```text
Initial hybrid collapsed to one class.
```

For the improved hybrid, mention:

```text
Best balanced performance after threshold tuning and stylometric fusion.
```

---

### 16. Update notebook markdown for paper story

Add clear markdown explaining the final narrative:

```markdown
The initial experiments showed that leakage-free detection is much harder than the original perfect scores suggested. Classical models and the initial hybrid model struggled after removing leakage. The final improved hybrid model combines DistilBERT semantic representations with stylometric writing signals and uses validation-based threshold tuning to avoid class collapse. This produced more balanced AI and human recall and became the most reliable model for the final paper.
```

Also add:

```markdown
The final model should not be interpreted as a perfect detector. Instead, it demonstrates that hybrid transformer–stylometric modeling improves robustness compared with simple baselines under a cleaner evaluation setting.
```

---

## Expected final result

After running the notebook in Colab, I should be able to say:

1. We audited and cleaned the dataset.
2. We compared classical, stylometric, transformer, and hybrid approaches.
3. The first hybrid model was unstable.
4. The improved hybrid model fixed the instability using:
   - mean pooling
   - stylometric fusion
   - validation threshold tuning
   - best-model saving
   - balanced metrics
5. The improved hybrid model is the best candidate for the final paper.

---

## Do not do this

- Do not run full training locally.
- Do not use leaking columns as features.
- Do not use `source`, `ai_model`, `id`, or `prompt`.
- Do not use the old raw dataset as the default.
- Do not remove the baseline comparison.
- Do not hide failed results.
- Do not claim 100% performance is valid.
- Do not leave the final hybrid using useless domain embedding.
- Do not evaluate using only accuracy.

---

## Final note

The final paper should present the improved hybrid model as the final proposed method, while the original notebook hybrid can be discussed as an initial failed attempt that motivated the improved version.
