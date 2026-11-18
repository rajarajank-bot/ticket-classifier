import time
import joblib
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score

# 1. Load Data
train_df = pd.read_csv("data/train.csv")
val_df = pd.read_csv("data/val.csv")
test_df = pd.read_csv("data/test.csv")

labels = sorted(train_df["category"].unique())
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for i, l in enumerate(labels)}

train_df["label"] = train_df["category"].map(label2id)
val_df["label"] = val_df["category"].map(label2id)
test_df["label"] = test_df["category"].map(label2id)

# 2. Measure Baseline Timing & Metrics on Test Set
baseline_cat = joblib.load("models/baseline_category.joblib")
t0 = time.time()
base_preds = baseline_cat.predict(test_df["text"])
base_inf_time_ms = ((time.time() - t0) / len(test_df)) * 1000
base_acc = accuracy_score(test_df["category"], base_preds)
base_f1 = f1_score(test_df["category"], base_preds, average="macro")

# 3. DistilBERT Fine-Tuning
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=64)

train_ds = Dataset.from_pandas(train_df[["text", "label"]]).map(tokenize_fn, batched=True)
val_ds = Dataset.from_pandas(val_df[["text", "label"]]).map(tokenize_fn, batched=True)
test_ds = Dataset.from_pandas(test_df[["text", "label"]]).map(tokenize_fn, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=len(labels), id2label=id2label, label2id=label2id
)

training_args = TrainingArguments(
    output_dir="./distilbert_out",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=10,
    report_to="none"
)

def compute_metrics(eval_pred):
    logits, y_true = eval_pred
    y_pred = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro")
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics
)

print("\n=== TRAINING DISTILBERT (2 EPOCHS) ===")
t_train_start = time.time()
trainer.train()
bert_train_time = time.time() - t_train_start

t_inf_start = time.time()
test_output = trainer.predict(test_ds)
bert_inf_time_ms = ((time.time() - t_inf_start) / len(test_df)) * 1000

bert_preds = np.argmax(test_output.predictions, axis=-1)
bert_acc = accuracy_score(test_df["label"], bert_preds)
bert_f1 = f1_score(test_df["label"], bert_preds, average="macro")

print("\n=== MODEL COMPARISON TABLE (TEST SET) ===")
print(f"| Model | Accuracy | Macro F1 | Training Time | Inference Time / Ticket |")
print(f"|---|---|---|---|---|")
print(f"| TF-IDF + LogisticRegression | {base_acc:.4f} | {base_f1:.4f} | ~0.5 sec | {base_inf_time_ms:.2f} ms |")
print(f"| Fine-tuned DistilBERT | {bert_acc:.4f} | {bert_f1:.4f} | {bert_train_time:.1f} sec | {bert_inf_time_ms:.2f} ms |")