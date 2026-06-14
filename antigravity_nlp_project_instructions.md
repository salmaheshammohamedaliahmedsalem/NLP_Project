# Antigravity Instructions — NLP Project Notebook

## Project Context
We are working on a university NLP project using the Kaggle dataset:

`ai_vs_human_content_v2_20000.csv`

The project topic is:

**Global AI vs. Human Content Detection (2026)**

The goal is to build a binary classifier that distinguishes between:

- Human-written content
- AI-generated content

The project must be paper-worthy and suitable for an IEEE-format technical report. The required rubric includes:

1. Related work / comparative study: implement at least 3 related works on the same dataset.
2. Proposed model with innovation.
3. Technical report in IEEE format, maximum 8 pages.
4. Presentation and oral discussion.
5. Deliverables: code, report, presentation.

We need the implementation in **notebook style**, with clear Markdown explanations, section headings, outputs, plots, and comparison tables.

---

## Important Requirement
Create a clean Jupyter notebook named:

`01_ai_vs_human_detection_project.ipynb`

The notebook should be readable as a mini research paper, not only code.

Use the following structure exactly.

---

# Notebook Structure

## 0. Project Title and Objective
Add Markdown explaining:

- Project title: **Domain-Aware Hybrid Transformer–Stylometric Detection of AI-Generated Content**
- Dataset: `ai_vs_human_content_v2_20000.csv`
- Task: binary classification: AI-generated vs human-written.
- Main research question:

> Can stylometric machine-signature features improve transformer-based AI text detection across multiple domains?

- Proposed model idea:

> A hybrid model that combines transformer embeddings with stylometric, readability, repetition, and probabilistic machine-signature features.

---

## 1. Setup and Imports
Create a setup cell that imports all necessary libraries.

Use:

- pandas
- numpy
- matplotlib
- seaborn only if already installed; otherwise use matplotlib only
- scikit-learn
- nltk or spaCy if available
- textstat if available, but code should not crash if unavailable
- transformers
- torch
- tqdm

The code should handle missing optional libraries gracefully.

Example behavior:

```python
try:
    import textstat
    HAS_TEXTSTAT = True
except ImportError:
    HAS_TEXTSTAT = False
    print("textstat not installed. Readability features will use fallback functions.")
```

---

## 2. Load Dataset
Load:

```python
DATA_PATH = "ai_vs_human_content_v2_20000.csv"
df = pd.read_csv(DATA_PATH)
```

Then print:

- shape
- columns
- first 5 rows
- missing values
- class distribution
- domain/category distribution if the dataset has a domain/category column

Important: inspect the actual column names first. Do not assume the exact names. Write code that identifies likely text, label, and domain columns.

Possible column detection logic:

```python
possible_text_cols = ["text", "content", "document", "article", "response"]
possible_label_cols = ["label", "generated", "is_ai", "class", "source"]
possible_domain_cols = ["domain", "category", "topic", "type"]
```

Then map them to:

```python
TEXT_COL
LABEL_COL
DOMAIN_COL
```

If automatic detection fails, raise a clear error telling the user which columns were found.

---

## 3. Data Cleaning and Label Encoding
Create clean columns:

```python
df["text_clean"]
df["label_binary"]
```

Requirements:

- Remove rows with missing text.
- Remove duplicate text rows.
- Keep cleaning light because stylometric features need punctuation and structure.
- Do NOT over-clean the text.
- Convert labels into binary format:
  - human = 0
  - AI = 1

Handle label formats robustly. The dataset may contain labels like:

- Human / AI
- human / ai
- 0 / 1
- model names such as GPT-4, Gemini, Claude vs Human

If the label column contains model names, treat anything not human as AI.

Add a Markdown explanation:

> We use light cleaning because aggressive preprocessing may destroy writing-style signals such as punctuation, sentence length, capitalization, and repetition.

---

## 4. Exploratory Data Analysis
Add EDA plots and tables:

1. Class distribution.
2. Domain distribution, if available.
3. Average text length by label.
4. Average word count by label.
5. Domain vs label distribution, if domain exists.

Create reusable helper functions:

```python
def count_words(text):
    return len(str(text).split())

def count_chars(text):
    return len(str(text))
```

Add interpretation Markdown after each major plot.

---

## 5. Train / Validation / Test Split
Use stratified splitting.

If domain column exists, try to preserve both label and domain by creating a stratification column:

```python
df["stratify_col"] = df["label_binary"].astype(str) + "_" + df[DOMAIN_COL].astype(str)
```

If some strata are too small, fall back to stratifying only by label.

Split:

- 70% train
- 15% validation
- 15% test

Create:

```python
train_df
val_df
test_df
```

Print class distribution for each split.

---

# Related Work Models / Baselines

We need at least 3 related-work implementations on the same dataset.

## 6. Baseline 1 — TF-IDF + Logistic Regression
Implement:

```python
TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)
```

Classifier:

```python
LogisticRegression(max_iter=2000, class_weight="balanced")
```

Evaluate on validation and test.

Save metrics:

- accuracy
- precision
- recall
- F1
- ROC-AUC if possible

---

## 7. Baseline 2 — TF-IDF + Linear SVM
Implement:

```python
LinearSVC(class_weight="balanced")
```

Use the same TF-IDF representation.

For ROC-AUC, use decision scores if possible.

Evaluate on validation and test.

---

## 8. Baseline 3 — Stylometric Features + Classical ML
Extract stylometric and linguistic machine-signature features.

Create a function:

```python
def extract_stylometric_features(text):
    ...
```

Features should include:

### Length features
- character count
- word count
- sentence count
- average word length
- average sentence length

### Punctuation features
- number of periods
- commas
- exclamation marks
- question marks
- semicolons
- colons
- quotation marks
- punctuation ratio

### Lexical diversity features
- unique word ratio
- type-token ratio
- hapax ratio if possible

### Repetition features
- repeated bigram ratio
- repeated trigram ratio
- most common word ratio

### Readability features
Use textstat if available:
- Flesch Reading Ease
- Flesch-Kincaid Grade
- SMOG Index

If textstat is unavailable, implement simple fallback readability proxies:
- average sentence length
- average word length
- long word ratio

### AI-signature probabilistic features
If feasible:
- perplexity using a small model such as `distilgpt2`
- burstiness as the standard deviation of sentence lengths divided by mean sentence length

If perplexity is too slow, make it optional:

```python
COMPUTE_PERPLEXITY = False
```

Default should be False to keep the notebook runnable quickly. Add a cell explaining how to enable it.

Classifiers to try:

1. RandomForestClassifier
2. XGBoost if installed; otherwise GradientBoostingClassifier
3. LogisticRegression

Use StandardScaler where needed.

Evaluate and save the best stylometric model.

---

## 9. Baseline 4 — Transformer Fine-Tuning
Implement a transformer baseline using HuggingFace.

Preferred model:

```python
MODEL_NAME = "roberta-base"
```

If compute is limited, allow:

```python
MODEL_NAME = "distilroberta-base"
```

Create Dataset class or use HuggingFace Dataset if installed.

Training settings:

```python
max_length = 256
batch_size = 8 or 16
epochs = 2 or 3
learning_rate = 2e-5
```

Use validation F1 for model selection.

Save:

- training loss
- validation metrics per epoch
- final test metrics

Also save the best transformer model to:

```python
models/transformer_baseline/
```

If full fine-tuning is too slow, add a fallback option:

- freeze transformer encoder
- train only classification head

But the notebook should be written to support full fine-tuning.

---

# Proposed Model

## 10. Proposed Model — Domain-Aware Hybrid Transformer–Stylometric Fusion
This is the main innovation.

Architecture:

```text
Input Text
   ↓
Transformer Encoder: RoBERTa / DistilRoBERTa
   ↓
[CLS] / pooled embedding

Stylometric Feature Extractor
   ↓
length + punctuation + lexical diversity + readability + repetition + burstiness

Optional Domain Embedding
   ↓
Domain/category embedding if dataset has a domain column

Fusion Layer
   ↓
concat(transformer_embedding, stylometric_vector, optional_domain_embedding)
   ↓
MLP classifier
   ↓
Human vs AI
```

Implementation requirements:

1. Extract stylometric features for all rows.
2. Scale stylometric features using StandardScaler fitted only on train data.
3. Create a PyTorch Dataset that returns:
   - tokenized input ids
   - attention mask
   - scaled stylometric feature vector
   - optional domain id
   - label
4. Build a PyTorch model class:

```python
class HybridTransformerStylometricModel(nn.Module):
    def __init__(self, transformer_name, stylometric_dim, num_domains=None):
        ...
```

5. Inside the model:
   - load AutoModel
   - get pooled representation using CLS token or last hidden state first token
   - pass stylometric features through a small dense layer
   - if domain exists, pass domain id through embedding layer
   - concatenate all representations
   - pass through MLP classifier

Suggested classifier head:

```python
nn.Sequential(
    nn.Linear(fusion_dim, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(64, 2)
)
```

Training:

- CrossEntropyLoss
- AdamW
- learning rate 2e-5 for transformer parameters
- learning rate 1e-3 for fusion/classifier parameters if using parameter groups
- 2 to 3 epochs
- early stopping based on validation F1

Save best model to:

```python
models/hybrid_fusion_model.pt
```

---

## 11. Ablation Study
This is extremely important for making the project paper-worthy.

Run and compare:

1. Transformer only
2. Stylometric only
3. Transformer + stylometric
4. Transformer + stylometric + domain embedding, if domain exists

Create a results table:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|

The ablation should answer:

> Do stylometric machine-signature features improve transformer detection?

---

## 12. Per-Domain Evaluation
If the dataset has a domain/category column, evaluate every model per domain.

Create a table like:

| Domain | Model | Accuracy | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|

Also create a bar chart showing F1 per domain.

This is one of the strongest parts of the project because the dataset is multi-domain.

Add discussion Markdown:

- Which domains are easiest?
- Which domains are hardest?
- Does the hybrid model help in difficult domains?
- Does code behave differently from natural language domains?

---

## 13. Error Analysis
Create an error analysis section.

For the proposed model:

1. Show false positives: human texts predicted as AI.
2. Show false negatives: AI texts predicted as human.
3. Show confidence scores if available.
4. Compare text length distribution for correct vs wrong predictions.
5. If domain exists, show which domains produce the most errors.

Add Markdown interpretation:

- False positives may happen when human text is formal, polished, repetitive, or template-like.
- False negatives may happen when AI text is creative, informal, or heavily varied.

---

## 14. Explainability
Add at least one explainability method.

For classical models:

- Show top TF-IDF features for Logistic Regression.
- Show feature importance for stylometric RandomForest/XGBoost.

For hybrid model:

- Show stylometric feature importance indirectly by training a simple model on the same features.
- Optional: use Integrated Gradients or SHAP only if easy and installed.

Minimum required:

1. Top words/features from TF-IDF Logistic Regression.
2. Top stylometric features from the stylometric model.

This is enough for report discussion.

---

## 15. Final Comparison Table
Create one final table with all models:

1. TF-IDF + Logistic Regression
2. TF-IDF + Linear SVM
3. Stylometric Features + ML
4. Transformer Baseline
5. Proposed Hybrid Model

Metrics:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC

Save the table to:

```python
results/final_results.csv
```

Also save per-domain results to:

```python
results/per_domain_results.csv
```

---

## 16. Report-Ready Figures
Save figures into a folder:

```python
figures/
```

Required figures:

1. Class distribution
2. Domain distribution
3. Text length by label
4. Final model comparison bar chart
5. Per-domain F1 comparison
6. Confusion matrix for proposed model
7. Proposed architecture diagram, if possible

---

## 17. Paper-Ready Conclusion Cell
Add a Markdown conclusion summarizing:

- Best performing model.
- Whether stylometric features improved performance.
- Which domains were hardest.
- Why hybrid detection is better than using only transformer models.
- Limitations.
- Future work.

Use wording similar to:

> The results show that combining transformer representations with stylometric machine-signature features provides a stronger and more interpretable detector than either feature family alone. The per-domain evaluation further demonstrates that AI-generated text detection is not uniform across domains, supporting the need for domain-aware evaluation.

---

# Coding Quality Requirements

Please make the notebook clean and robust.

Use:

```python
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
```

Create folders automatically:

```python
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)
```

Use reusable functions for:

- metric calculation
- plotting confusion matrix
- extracting stylometric features
- model evaluation
- per-domain evaluation

Do not hard-code dataset columns without inspecting them first.

---

# Expected Output Files

By the end, the project directory should contain:

```text
NLP_Project/
│
├── ai_vs_human_content_v2_20000.csv
├── 01_ai_vs_human_detection_project.ipynb
│
├── models/
│   ├── transformer_baseline/
│   └── hybrid_fusion_model.pt
│
├── results/
│   ├── final_results.csv
│   ├── per_domain_results.csv
│   └── training_logs.csv
│
└── figures/
    ├── class_distribution.png
    ├── domain_distribution.png
    ├── text_length_by_label.png
    ├── model_comparison.png
    ├── per_domain_f1.png
    ├── proposed_confusion_matrix.png
    └── proposed_architecture.png
```

---

# Research Framing to Use in Markdown

Use these terms throughout the notebook:

- AI-generated text detection
- human-authored text
- stylometric features
- machine signatures
- domain-aware evaluation
- transformer embeddings
- hybrid feature fusion
- per-domain robustness
- ablation study
- interpretability

Avoid making unsupported claims like “perfect detection.” Instead, use academic phrasing:

- “The results suggest...”
- “The hybrid model improves...”
- “The domain-wise results indicate...”
- “This supports the hypothesis that...”

---

# Important Notes

1. Keep the notebook runnable from top to bottom.
2. Use comments and Markdown explanations heavily.
3. The notebook should be understandable by a professor reviewing the project.
4. Focus on clean implementation and strong experimental comparison.
5. The proposed model must not be only a fine-tuned transformer. The innovation is the fusion of transformer embeddings with stylometric machine-signature features and optional domain information.
