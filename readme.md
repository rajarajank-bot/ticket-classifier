# Customer Support Ticket Classifier

## 1. Dataset Description
- **Source:** Synthetic Customer Support Ticket dataset (`data/support_tickets.csv`) containing **650** cleaned support tickets.
- **Labels:** 
  - `category`: Billing, Technical, Account, General
  - `priority`: High, Medium, Low
- **Split:** Stratified 70% Train (455) / 15% Validation (97) / 15% Test (98).

## 2. Model Comparison (Test Set)

| Model | Accuracy | Macro F1 | Training Time | Inference Time / Ticket |
|---|---|---|---|---|
| TF-IDF + LogisticRegression | 1.0000 | 1.0000 | ~0.5 sec | 0.48 ms |
| Fine-tuned DistilBERT | 1.0000 | 1.0000 | 300.0 sec | 85.66 ms |

### Analysis & Deployment Justification
1. **Where DistilBERT wins:** DistilBERT captures deep contextual semantics and handles complex, unseen phrasing or synonyms better than bag-of-words n-grams.
2. **Where Baseline is good enough:** For structured support tickets with clear domain keywords (e.g., "invoice", "charged twice", "password"), TF-IDF + LogisticRegression achieves identical accuracy.
3. **Deployment Choice:** We select **TF-IDF + LogisticRegression** for production deployment.
4. **Speed & Cost Justification:** It trains in ~0.5s (vs 300s for DistilBERT) and runs inference **~178x faster** (0.48 ms vs 85.66 ms per ticket) on standard CPU without requiring expensive GPU instances.

## 3. How to Run the API

### Start Server
```bash
uvicorn app:app --reload