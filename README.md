# spam-seeker

A Python-based spam email detection AI agent using **scikit-learn** and **MLPClassifier**, with optional integration as a **Camunda Zeebe worker** for workflow automation. 

## Features

- Train a spam detection model using TF-IDF and a simple neural network.
- Evaluate the model on a test dataset.
- Predict whether a single email is **spam** or **ham**.
- Run as a **Camunda Zeebe worker** to handle automated email classification tasks.

## Requirements

Python 3.8+ and the following packages:
```text
scikit-learn
grpcio
grpcio-tools
pyzeebe
pandas
datasets
logging
```
### Install dependencies with:

`pip install -r requirements.txt`

### Usage

Run the main script with one of the following options:

1. Train the model `python spam-seeker.py --train`
2. Evaluate the model `python spam-seeker.py --evaluate`
3. Predict a single email `python spam-seeker.py --predict "Your email text here"`
4. Run as Camunda Zeebe worker `python spam-seeker.py --camunda-worker`

### Deploy alongside Camunda 8

You can also deploy this project alongside [Camunda 8](https://camunda.com/platform/) in a Dockerized environment.

`cd docker && docker compose up -d`

Once all containers are succesfully running you have the possiblity to deploy a workflow example plus the related form in the `bpmn/` folder.

<img src="https://github.com/fjperezcostas/spam-seeker/blob/master/bpmn/bpmn-workflow.png" />

### Model Details
- Vectorizer: **TfidfVectorizer**
- Classifier: **MLPClassifier** with:
	- Hidden layers: (8, 8)
	- Activation: relu
	- Solver: adam
	- Max iterations: 200

### Evaluation Report

```text
[INFO] Starting spam-seeker with dataset **mshenoda/spam-messages**
[INFO] Loading 'validation' split of dataset...
[INFO] Starting model evaluation...
[INFO] Number of test samples: 5923
[INFO] Overall Test Accuracy: **0.9568**
[INFO] Detailed Classification Report:
[INFO]               precision    recall  f1-score   support

         ham       0.97      0.96      0.96      3476
        spam       0.94      0.95      0.95      2447

    accuracy                           0.96      5923
   macro avg       0.95      0.96      0.96      5923
weighted avg       0.96      0.96      0.96      5923

[INFO] Evaluation complete.
```