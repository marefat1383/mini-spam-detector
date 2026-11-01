# Mini Spam Detector

A small educational AI project that classifies short messages as `spam` or `not spam` using TF-IDF features and Logistic Regression from scikit-learn.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Run

```bash
python spam_detector.py "Congratulations, you won a free prize!"
python spam_detector.py "Can we meet for lunch tomorrow?"
```

## Test

```bash
python -m unittest test_spam_detector.py
```

The model uses a tiny in-code dataset for demonstration and is not intended for production filtering.
