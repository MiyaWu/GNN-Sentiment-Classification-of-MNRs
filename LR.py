import logging
import torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(filename='logistic_regression_experiment1.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
combined_vectors = np.loadtxt("tfidf_doc2vec_features.csv", delimiter=",")
y = torch.tensor(np.loadtxt("labels.csv", delimiter=","), dtype=torch.long)

X = combined_vectors
y = y.numpy()

test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]
num_experiments = 10

for test_size in test_sizes:
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    for experiment in range(1, num_experiments + 1):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        lr_model = LogisticRegression(max_iter=1000)
        lr_model.fit(X_train, y_train)
        y_pred = lr_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

        precision = precision_score(y_test, y_pred, average='weighted')
        precisions.append(precision)

        recall = recall_score(y_test, y_pred, average='weighted')
        recalls.append(recall)

        f1 = f1_score(y_test, y_pred, average='weighted')
        f1_scores.append(f1)

    average_accuracy = np.mean(accuracies)
    average_precision = np.mean(precisions)
    average_recall = np.mean(recalls)
    average_f1 = np.mean(f1_scores)

    logging.info(f'\nAverage Test Accuracy (LR) over {num_experiments} experiments: {average_accuracy * 100:.2f}%')
    logging.info(f'Average Precision (LR) over {num_experiments} experiments: {average_precision * 100:.2f}%')
    logging.info(f'Average Recall (LR) over {num_experiments} experiments: {average_recall * 100:.2f}%')
    logging.info(f'Average F1 Score (LR) over {num_experiments} experiments: {average_f1 * 100:.2f}%')
