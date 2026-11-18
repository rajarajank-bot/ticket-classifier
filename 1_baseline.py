import os
import random
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ---------------------------------------------------------
# TASK 1: Dataset Preparation (650 Support Tickets)
# ---------------------------------------------------------
random.seed(42)
np.random.seed(42)

templates = [
    ("I was charged twice for my monthly subscription", "Billing", "High"),
    ("My invoice shows an incorrect tax amount", "Billing", "Medium"),
    ("Need a refund for the cancelled order immediately", "Billing", "High"),
    ("How do I download my past billing receipts?", "Billing", "Low"),
    ("Payment failed using my credit card", "Billing", "High"),
    ("Update my billing address on the account", "Billing", "Low"),
    ("The mobile app crashes every time I open settings", "Technical", "High"),
    ("Server error 500 when trying to export data", "Technical", "High"),
    ("Dashboard is loading very slowly today", "Technical", "Medium"),
    ("API webhook is not triggering on new events", "Technical", "High"),
    ("Dark mode toggle is not saving my preference", "Technical", "Low"),
    ("Cannot upload PDF files larger than 5MB", "Technical", "Medium"),
    ("I forgot my password and reset link expired", "Account", "High"),
    ("Locked out of my account after two-factor authentication", "Account", "High"),
    ("How to change my registered email address?", "Account", "Medium"),
    ("Please delete my account and personal data", "Account", "Medium"),
    ("Want to upgrade my plan to enterprise tier", "Account", "Low"),
    ("Add a new team member to my workspace", "Account", "Low"),
    ("What are your customer support working hours?", "General", "Low"),
    ("Do you offer discounts for non-profit organizations?", "General", "Low"),
    ("Where can I find the user documentation?", "General", "Low"),
    ("Feature request: add dark theme to reports", "General", "Low"),
    ("Need clarification on your privacy policy", "General", "Medium"),
    ("Great service, just wanted to share feedback", "General", "Low"),
]

noise_words = ["please help", "urgent", "thanks", "asap", "kindly check", "hello team", "issue since morning", "not working"]

records = []
for i in range(650):
    text, cat, prio = random.choice(templates)
    prefix = random.choice(["", "Hi, ", "Hello, ", "Urgent: ", "Support needed - "])
    suffix = f" ({random.choice(noise_words)} #{i})"
    records.append({
        "text": f"{prefix}{text}{suffix}",
        "category": cat,
        "priority": prio
    })

df = pd.DataFrame(records)
df["text"] = df["text"].str.lower().str.strip()
df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

os.makedirs("data", exist_ok=True)
df.to_csv("data/support_tickets.csv", index=False)

print("=== TASK 1: DATASET SUMMARY ===")
print(f"Total clean rows: {len(df)}")
print("\nCategory Class Balance:\n", df["category"].value_counts())
print("\nPriority Class Balance:\n", df["priority"].value_counts())

df["strat_key"] = df["category"] + "_" + df["priority"]
train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["strat_key"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["strat_key"], random_state=42)

print(f"\nSplit Sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
train_df.to_csv("data/train.csv", index=False)
val_df.to_csv("data/val.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

# ---------------------------------------------------------
# TASK 2: Baseline Model (TF-IDF + Logistic Regression)
# ---------------------------------------------------------
def train_and_evaluate(target_col):
    print(f"\n=== TASK 2: BASELINE MODEL FOR '{target_col.upper()}' ===")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])

    param_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__C": [0.1, 1.0, 10.0]
    }

    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring="f1_macro", n_jobs=-1)
    grid.fit(train_df["text"], train_df[target_col])

    best_model = grid.best_estimator_
    print("Best Params:", grid.best_params_)

    preds = best_model.predict(val_df["text"])
    probs = best_model.predict_proba(val_df["text"])
    max_probs = np.max(probs, axis=1)

    acc = accuracy_score(val_df[target_col], preds)
    macro_f1 = f1_score(val_df[target_col], preds, average="macro")
    cm = confusion_matrix(val_df[target_col], preds)

    print(f"Validation Accuracy: {acc:.4f} | Macro F1: {macro_f1:.4f}")
    print("Confusion Matrix:\n", cm)

    val_eval = val_df.copy()
    val_eval["pred"] = preds
    val_eval["confidence"] = max_probs
    confusing = val_eval.sort_values(by="confidence", ascending=True).head(10)

    print(f"\nTop 10 Most Confusing Examples ({target_col}):")
    for idx, row in confusing.iterrows():
        print(f"- Text: '{row['text'][:55]}...' | Actual: {row[target_col]} | Pred: {row['pred']} | Conf: {row['confidence']:.2f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, f"models/baseline_{target_col}.joblib")
    return best_model

train_and_evaluate("category")
train_and_evaluate("priority")
print("\n[SUCCESS] First Half Complete! Models saved in 'models/' folder.")