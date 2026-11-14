# Credit Card Fraud Detection

This project detects fraudulent credit card transactions with machine learning
and compares multiple strategies for handling severe class imbalance.

## Dataset

The project uses the [Credit Card Fraud Detection dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
It contains anonymized transaction features, `Time`, `Amount`, and the binary
target column `Class`:

- `Class = 0`: legitimate transaction
- `Class = 1`: fraudulent transaction

The dataset is not included in this repository. Download `creditcard.csv` from
Kaggle and place it in the project root.

## Installation

```bash
git clone https://github.com/Chen729689OUGK/credit-card-fraud-detection.git
cd credit-card-fraud-detection

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Training and evaluation

Run the complete experiment:

```bash
python credit_card_fraud.py
```

The script uses the same stratified train/test split for three Random Forest
strategies:

1. Standard class weighting
2. `class_weight="balanced"`
3. SMOTE applied only to the training folds

It reports precision, recall, F1-score, PR-AUC, ROC-AUC, and confusion matrices.
It also runs five-fold stratified cross-validation on the training data before
the final test evaluation.

Optional arguments:

```bash
python credit_card_fraud.py --threshold 0.3
python credit_card_fraud.py --cv-folds 3
python credit_card_fraud.py --plot
python credit_card_fraud.py --data /path/to/creditcard.csv
python credit_card_fraud.py --model-output models/fraud_model.joblib
```

The probability threshold should be selected according to the operational cost
of missed fraud versus false alerts. PR-AUC is the primary comparison metric
because the fraud class is highly imbalanced. The model with the best
cross-validation PR-AUC is saved as a Joblib file by default at
`fraud_detection_model.joblib`.

## Tests

Run the automated tests:

```bash
pytest -q
```

The tests cover dataset validation, stratified splitting, and probability
generation from the SMOTE pipeline.

## Project structure

```text
credit_card_fraud.py       # Training and evaluation pipeline
tests/                     # Automated tests
credit_card_fraud.ipynb    # Exploratory notebook
requirements.txt           # Python dependencies
fraud_detection_model.joblib # Generated model artifact (ignored by Git)
```

## Limitations

- The dataset contains anonymized features and may not represent a production
  transaction stream.
- The default threshold of `0.5` is not calibrated to a specific business cost.
- Model performance should be re-evaluated on current, representative data
  before deployment.
