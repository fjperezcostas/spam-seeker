import asyncio
import csv
import pickle
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from pyzeebe import ZeebeWorker, create_insecure_channel

def train(train_texts, train_labels, model_path="models/model.pkl", vectorizer_path="models/vectorizer.pkl"):
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_texts)

    model = MLPClassifier(
        hidden_layer_sizes=(8, 8),
        activation='relu',
        solver='sgd',
        max_iter=1000,
        verbose=True
    )
    model.fit(X_train, train_labels)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"Training complete. Model saved to {model_path}, vectorizer saved to {vectorizer_path}")
    return model, vectorizer


def predict(text, model_path="models/model.pkl", vectorizer_path="models/vectorizer.pkl"):
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    X_test = vectorizer.transform([text])
    probability = model.predict_proba(X_test)[0][1]
    label = "spam" if probability > 0.5 else "ham"

    return label, probability


def evaluate(test_texts, test_labels, model_path="models/model.pkl", vectorizer_path="models/vectorizer.pkl"):
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    X_test = vectorizer.transform(test_texts)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(test_labels, y_pred)
    print(f"Test accuracy: {accuracy:.4f}")


async def start_worker():
    channel = create_insecure_channel(grpc_address="zeebe:26500")
    worker = ZeebeWorker(channel)

    @worker.task(task_type="check-spam")
    def check_spam(email: str) -> dict:
        label, probability = predict(email)
        return {"label": label, "probability": probability}

    print("Camunda worker started...")
    await worker.work()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spam Email Detector (only scikit-learn)")
    parser.add_argument("--train", action="store_true", help="Train the model using train.csv")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model using test.csv")
    parser.add_argument("--predict", type=str, help="Predict SPAM/HAM")
    parser.add_argument("--camunda-worker", action="store_true", help="Run worker")
    args = parser.parse_args()

    if args.train:
        train_texts, train_labels = [], []
        with open("data/train.csv", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                train_texts.append(row["text"])
                train_labels.append(int(row["label"]))

        train(train_texts, train_labels)

    elif args.evaluate:
        test_texts, test_labels = [], []
        with open("data/test.csv", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                test_texts.append(row["text"])
                test_labels.append(int(row["label"]))

        evaluate(test_texts, test_labels)

    elif args.predict:
        label, prob = predict(args.predict)
        print(f"{args.predict} → {label} ({prob:.2f})")

    elif args.camunda_worker:
        asyncio.run(start_worker())

    else:
        print("Use --train, --evaluate, --predict or --camunda-worker")
