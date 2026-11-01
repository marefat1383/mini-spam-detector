import argparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

TRAINING_MESSAGES = [
    "Win a free vacation now",
    "Claim your prize by clicking this link",
    "Exclusive offer just for you",
    "Congratulations you won cash",
    "Limited time free gift",
    "Can we meet for lunch tomorrow",
    "Please send me the project notes",
    "Your appointment is at three pm",
    "Happy birthday see you tonight",
    "Could you call me after work",
]
TRAINING_LABELS = ["spam"] * 5 + ["not spam"] * 5


def build_model() -> Pipeline:
    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])
    model.fit(TRAINING_MESSAGES, TRAINING_LABELS)
    return model


def predict(message: str) -> str:
    return str(build_model().predict([message])[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a message as spam or not spam.")
    parser.add_argument("message", nargs="+", help="Message to classify")
    args = parser.parse_args()
    print(f"Prediction: {predict(' '.join(args.message))}")


if __name__ == "__main__":
    main()
