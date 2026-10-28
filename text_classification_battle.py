# %% [markdown]
# # Text Classification Battle: Decision Tree vs KNN vs SVM vs Naive Bayes
# Course: 21CSE377T Information Retrieval, CLA-2 Task 2
# Dataset: 20 Newsgroups (5-class subset) loaded through scikit-learn
#
# HOW TO RUN (VS Code): open this file, click "Run Cell" above each "# %%" cell
# in order, or run everything with:  python text_classification_battle.py
# Every number you report must come from YOUR run of this file.


import os
import re
import time
import random
from functools import lru_cache

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import (
    TfidfVectorizer, CountVectorizer, ENGLISH_STOP_WORDS,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, ConfusionMatrixDisplay,
)
from nltk.stem import PorterStemmer  # works offline, no extra downloads needed

SEED = 42                      # fixed seed so results are reproducible
random.seed(SEED)
np.random.seed(SEED)

OUT_DIR = "results"            # all tables and figures are saved here
os.makedirs(OUT_DIR, exist_ok=True)
print("scikit-learn version:", sklearn.__version__)

# %% 1. Define the problem and load the dataset
# Problem: given a post, predict which of 5 topics (newsgroups) it belongs to.
CATEGORIES = [
    "comp.graphics",
    "rec.autos",
    "sci.med",
    "talk.politics.guns",
    "soc.religion.christian",
]

# remove=(headers, footers, quotes) strips metadata such as email headers.
# Without this, models can "cheat" by reading the newsgroup name in the header.
raw = fetch_20newsgroups(
    subset="all",
    categories=CATEGORIES,
    remove=("headers", "footers", "quotes"),
    shuffle=True,
    random_state=SEED,
)

df = pd.DataFrame({
    "text": raw.data,
    "label": [raw.target_names[i] for i in raw.target],
})
df["text"] = df["text"].str.strip()
n_loaded = len(df)
df = df[df["text"].str.len() > 0].reset_index(drop=True)   # drop empty posts
df.insert(0, "doc_id", range(1, len(df) + 1))
print(f"Loaded {n_loaded} documents, kept {len(df)} non-empty documents.")

df.to_csv(os.path.join(OUT_DIR, "dataset_raw.csv"), index=False)

# %% 2. Inspect the data: size, labels, class balance
counts = df["label"].value_counts().sort_index()
summary = pd.DataFrame({
    "documents": counts,
    "percent": (counts / len(df) * 100).round(2),
})
print(summary)
print("\nTotal documents:", len(df))
print("Number of classes:", df["label"].nunique())
print("Imbalance ratio (largest / smallest class): "
      f"{counts.max() / counts.min():.2f}")

words_per_doc = df["text"].str.split().str.len()
print("\nWords per document (raw text):")
print(words_per_doc.describe().round(1))

summary.to_csv(os.path.join(OUT_DIR, "class_distribution.csv"))

ax = counts.plot(kind="bar", color="steelblue", figsize=(8, 4))
ax.set_title("Class distribution")
ax.set_ylabel("Number of documents")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "class_distribution.png"), dpi=150)
plt.show()

# %% 3. Preprocessing: lowercase, tokenize, stop-word removal, stemming
# This step does not learn anything from the data (no fitting), so doing it
# on all documents before the split does NOT cause data leakage.
stemmer = PorterStemmer()
STOP_WORDS = set(ENGLISH_STOP_WORDS)


@lru_cache(maxsize=None)
def stem_word(word):
    return stemmer.stem(word)


def preprocess(text):
    tokens = re.findall(r"[a-z]+", text.lower())                 # lowercase + tokenize
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]  # stop words
    return " ".join(stem_word(t) for t in tokens)                # stemming


df["clean_text"] = df["text"].apply(preprocess)

n_before = len(df)
df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
print(f"Dropped {n_before - len(df)} documents that became empty after cleaning.")
print("Documents remaining:", len(df))

# Before / after example (use this in your report)
example = df.iloc[0]
print("\n--- BEFORE (first 400 characters) ---")
print(example["text"][:400])
print("\n--- AFTER (first 400 characters) ---")
print(example["clean_text"][:400])
print("\nTokens before:", len(example["text"].split()),
      "| tokens after:", len(example["clean_text"].split()))

df.to_csv(os.path.join(OUT_DIR, "dataset_preprocessed.csv"), index=False)

# %% 4. Reproducible stratified train/test split (80/20)
train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=SEED,
)
X_train, y_train = train_df["clean_text"], train_df["label"]
X_test, y_test = test_df["clean_text"], test_df["label"]

print("Train size:", len(train_df), "| Test size:", len(test_df))
split_table = pd.DataFrame({
    "train": y_train.value_counts().sort_index(),
    "test": y_test.value_counts().sort_index(),
})
print(split_table)
split_table.to_csv(os.path.join(OUT_DIR, "split_distribution.csv"))

class_names = sorted(df["label"].unique())

# %% 5. Models: same TF-IDF features for all four classifiers
# TF-IDF is inside the Pipeline, so it is fitted ONLY on training data.


def make_vectorizer():
    return TfidfVectorizer(min_df=2, max_features=5000, sublinear_tf=True)


def make_models():
    return {
        "Decision Tree": DecisionTreeClassifier(random_state=SEED),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5, metric="cosine"),
        "SVM (linear)": LinearSVC(random_state=SEED, max_iter=5000),
        "Naive Bayes": MultinomialNB(),
    }


def make_pipeline(model):
    return Pipeline([("tfidf", make_vectorizer()), ("clf", model)])


# %% 6. Train, predict, and compute metrics from actual predictions
def evaluate(pipe):
    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = pipe.predict(X_test)
    predict_time = time.perf_counter() - t0

    p, r, f, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0)
    n_features = len(pipe.named_steps["tfidf"].vocabulary_)

    row = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision (macro)": p, "Recall (macro)": r, "F1 (macro)": f,
        "Precision (weighted)": pw, "Recall (weighted)": rw, "F1 (weighted)": fw,
        "Features": n_features,
        "Train time (s)": fit_time,
        "Predict time (s)": predict_time,
    }
    return row, y_pred


results, predictions, fitted = {}, {}, {}
for name, model in make_models().items():
    pipe = make_pipeline(model)
    results[name], predictions[name] = evaluate(pipe)
    fitted[name] = pipe
    print(f"Done: {name}")

results_df = pd.DataFrame(results).T
results_df["Features"] = results_df["Features"].astype(int)
print("\n=== TEST SET RESULTS (macro and weighted averages) ===")
print(results_df.round(4).to_string())
results_df.round(4).to_csv(os.path.join(OUT_DIR, "comparison_table.csv"))

# Extra evidence of fairness: 5-fold stratified CV on the TRAINING data only
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_rows = {}
for name, model in make_models().items():
    scores = cross_validate(
        make_pipeline(model), X_train, y_train, cv=cv,
        scoring=["accuracy", "f1_macro"],
    )
    cv_rows[name] = {
        "CV accuracy (mean)": scores["test_accuracy"].mean(),
        "CV accuracy (std)": scores["test_accuracy"].std(),
        "CV F1 macro (mean)": scores["test_f1_macro"].mean(),
        "CV F1 macro (std)": scores["test_f1_macro"].std(),
    }
cv_df = pd.DataFrame(cv_rows).T
print("\n=== 5-FOLD CROSS-VALIDATION ON TRAINING DATA ===")
print(cv_df.round(4).to_string())
cv_df.round(4).to_csv(os.path.join(OUT_DIR, "cross_validation.csv"))

# %% 7. Per-class report and confusion matrices
for name in results:
    print(f"\n===== {name} =====")
    print(classification_report(y_test, predictions[name],
                                target_names=class_names, zero_division=0))

# All four confusion matrices in one figure
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, name in zip(axes.ravel(), results):
    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions[name], labels=class_names, ax=ax,
        xticks_rotation=40, colorbar=False, cmap="Blues",
    )
    ax.set_title(name)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrices_all.png"), dpi=150)
plt.show()

# Best model (by macro F1) with normalized matrix
best_name = results_df["F1 (macro)"].astype(float).idxmax()
print("Best model by macro F1:", best_name)
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay.from_predictions(
    y_test, predictions[best_name], labels=class_names, ax=ax,
    xticks_rotation=40, normalize="true", values_format=".2f", cmap="Blues",
)
ax.set_title(f"{best_name}: normalized confusion matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix_best.png"), dpi=150)
plt.show()

# %% 8. Comparison graphs
metric_cols = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"]
ax = results_df[metric_cols].astype(float).plot(
    kind="bar", figsize=(10, 5), ylim=(0, 1))
ax.set_title("Classifier comparison (test set)")
ax.set_ylabel("Score")
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "comparison_metrics.png"), dpi=150)
plt.show()

ax = results_df[["Train time (s)", "Predict time (s)"]].astype(float).plot(
    kind="bar", figsize=(8, 4))
ax.set_title("Computational time")
ax.set_ylabel("Seconds")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "comparison_time.png"), dpi=150)
plt.show()

# %% 9. What-if experiment 1: change k in KNN
# TF-IDF is fitted on training data only, then reused for every k.
# Note: this is an analysis experiment. Do not tune k on the test set and then
# report that tuned score as your "fair" final result.
tfidf = make_vectorizer().fit(X_train)
Xtr, Xte = tfidf.transform(X_train), tfidf.transform(X_test)

knn_rows = []
for k in [1, 3, 5, 7, 11, 15, 25]:
    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine").fit(Xtr, y_train)
    pred = knn.predict(Xte)
    _, _, f, _ = precision_recall_fscore_support(
        y_test, pred, average="macro", zero_division=0)
    knn_rows.append({"k": k, "Accuracy": accuracy_score(y_test, pred),
                     "F1 (macro)": f})
knn_df = pd.DataFrame(knn_rows)
print("\n=== WHAT-IF 1: KNN with different k ===")
print(knn_df.round(4).to_string(index=False))
knn_df.round(4).to_csv(os.path.join(OUT_DIR, "whatif_knn_k.csv"), index=False)

plt.figure(figsize=(7, 4))
plt.plot(knn_df["k"], knn_df["Accuracy"], marker="o", label="Accuracy")
plt.plot(knn_df["k"], knn_df["F1 (macro)"], marker="s", label="F1 (macro)")
plt.xlabel("k (number of neighbours)")
plt.ylabel("Score")
plt.title("What-if: effect of k on KNN")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "whatif_knn_k.png"), dpi=150)
plt.show()

# %% 10. What-if experiment 2: change the feature representation
# Raw word counts vs TF-IDF unigrams vs TF-IDF unigrams + bigrams, all 4 models.
representations = {
    "Raw counts": lambda: CountVectorizer(min_df=2, max_features=5000),
    "TF-IDF (unigram)": lambda: TfidfVectorizer(
        min_df=2, max_features=5000, sublinear_tf=True),
    "TF-IDF (uni+bigram)": lambda: TfidfVectorizer(
        min_df=2, max_features=5000, sublinear_tf=True, ngram_range=(1, 2)),
}

rep_rows = []
for rep_name, make_vec in representations.items():
    for model_name, model in make_models().items():
        pipe = Pipeline([("vec", make_vec()), ("clf", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        _, _, f, _ = precision_recall_fscore_support(
            y_test, pred, average="macro", zero_division=0)
        rep_rows.append({"Representation": rep_name, "Model": model_name,
                         "Accuracy": accuracy_score(y_test, pred),
                         "F1 (macro)": f})
    print(f"Done: {rep_name}")

rep_df = pd.DataFrame(rep_rows)
rep_pivot = rep_df.pivot(index="Model", columns="Representation",
                         values="F1 (macro)")
print("\n=== WHAT-IF 2: macro F1 by feature representation ===")
print(rep_pivot.round(4).to_string())
rep_df.round(4).to_csv(os.path.join(OUT_DIR, "whatif_representation.csv"),
                       index=False)

ax = rep_pivot.plot(kind="bar", figsize=(10, 5), ylim=(0, 1))
ax.set_title("What-if: feature representation vs macro F1")
ax.set_ylabel("F1 (macro)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "whatif_representation.png"), dpi=150)
plt.show()

# %% 11. Wrap-up
print("\nAll outputs saved in the folder:", os.path.abspath(OUT_DIR))
print("Now paste the printed tables back to get help interpreting them.")
