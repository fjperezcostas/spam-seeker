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
```
### Install dependencies with:

`pip install -r requirements.txt`

### Usage

Run the main script with one of the following options:

1. Train the model `python spam_detector.py --train`
2. Evaluate the model `python spam_detector.py --evaluate`
3. Predict a single email `python spam_detector.py --predict "Your email text here"`
4. Run as Camunda Zeebe worker `python spam_detector.py --camunda-worker`

### Deploy alongside Camunda 8

You can also deploy this project alongside [Camunda 8](https://camunda.com/platform/) in a Dockerized environment.

`cd docker && docker compose up -d`

Once all containers are succesfully deployed you have the possiblity to deploy a workflow example plus the related form in the `bpmn/` folder.

### Model Details
- Vectorizer: **TfidfVectorizer**
- Classifier: **MLPClassifier** with:
	- Hidden layers: (8, 8)
	- Activation: relu
	- Solver: sgd
	- Max iterations: 1000
