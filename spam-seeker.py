import asyncio
import csv
import pickle
import argparse
import logging
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from pyzeebe import ZeebeWorker, create_insecure_channel

logger = logging.getLogger('spam-seeker')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s] %(message)s')
ch.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(ch)

def train(train_texts, train_labels, model_path="models/model.pkl", vectorizer_path="models/vectorizer.pkl"):
    logger.info("Starting model training...")
    logger.info(f"Number of training samples: {len(train_texts)}")
    vectorizer = TfidfVectorizer() 
    X_train = vectorizer.fit_transform(train_texts)
    logger.info(f"TF-IDF Vectorizer fitted. Feature count: {X_train.shape[1]}")
    model = MLPClassifier(
        hidden_layer_sizes=(8,8),
        verbose=True
    )
    logger.info("Training MLPClassifier...")
    model.fit(X_train, train_labels)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
    logger.info(f"Training complete. Model saved to **{model_path}**, vectorizer saved to **{vectorizer_path}**")
    return model, vectorizer

def predict(text, model_path="models/model.pkl", vectorizer_path="models/vectorizer.pkl"):
    logger.info(f"Loading model and vectorizer from **{model_path}** and **{vectorizer_path}**...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    X_test = vectorizer.transform([text])
    probability = model.predict_proba(X_test)[0][1] 
    label = "spam" if probability > 0.5 else "ham"
    logger.info("Prediction executed.")
    return label, probability

def evaluate(test_texts, test_labels, model_path="models/model.pkl", vectorizer_path="models/vectorizer.pkl"):
    logger.info("Starting model evaluation...")
    logger.info(f"Number of test samples: {len(test_texts)}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    X_test = vectorizer.transform(test_texts)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(test_labels, y_pred)
    logger.info(f"Overall Test Accuracy: **{accuracy:.4f}**")
    logger.info("Detailed Classification Report:")
    report = classification_report(test_labels, y_pred, target_names=['ham', 'spam'])
    logger.info(report)
    logger.info("Evaluation complete.")

async def start_worker():
    logger.info("Connecting to Zeebe...")
    channel = create_insecure_channel(grpc_address="zeebe:26500") 
    worker = ZeebeWorker(channel)
    @worker.task(task_type="check-spam")
    def check_spam(email: str) -> dict:
        logger.info(f"\nReceived task 'check-spam' for email: '{email[:50]}...'")
        label, probability = predict(email)
        logger.info(f"Result: {label.upper()} (Prob: {probability:.2f})")
        return {"label": label, "probability": probability}
    logger.info("Camunda worker started and listening for tasks on check-spam...")
    await worker.work()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="spam-seeker AI model trained for detecting spam emails/messages")
    parser.add_argument("--train", action="store_true", help="trains the model")
    parser.add_argument("--evaluate", action="store_true", help="evaluates the model")
    parser.add_argument("--predict", type=str, help="given a certain message, it predicts if it is spam/ham")
    parser.add_argument("--camunda-worker", action="store_true", help="Camunda worker")
    args = parser.parse_args()
    DATASET_NAME = "mshenoda/spam-messages"
    logger.info(f"Starting spam-seeker with dataset **{DATASET_NAME}**")
    if args.train:
        logger.info("Loading 'train' split of dataset...")
        ds_train = load_dataset(DATASET_NAME, split='train')
        df_train = ds_train.to_pandas()
        train_texts = df_train["text"].tolist()
        train_labels = df_train["label"].map({'ham': 0, 'spam': 1}).tolist() 
        train(train_texts, train_labels)
    elif args.evaluate:
        logger.info("Loading 'validation' split of dataset...")
        ds_test = load_dataset(DATASET_NAME, split='validation')
        df_test = ds_test.to_pandas()
        test_texts = df_test["text"].tolist()
        test_labels = df_test["label"].map({'ham': 0, 'spam': 1}).tolist()
        evaluate(test_texts, test_labels)
    elif args.predict:
        label, prob = predict(args.predict)
        logger.info(f"Input:  '{args.predict}'")
        logger.info(f"Result: **{label.upper()}** ({prob:.2f})")
    elif args.camunda_worker:
        asyncio.run(start_worker())
    else:
        logger.error("No valid argument provided.")
        logger.info("Usage: --train, --evaluate, --predict 'text' or --camunda-worker")
