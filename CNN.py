import logging
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

logging.basicConfig(filename='cnn_experiment.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

combined_vectors = np.loadtxt("tfidf_doc2vec_features.csv", delimiter=",")
labels = torch.tensor(np.loadtxt("labels.csv", delimiter=","), dtype=torch.long)

scaler = StandardScaler()
combined_vectors = scaler.fit_transform(combined_vectors)

features = torch.tensor(combined_vectors, dtype=torch.float32)
labels = torch.tensor(labels.numpy(), dtype=torch.long)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# features = features.to(device)
# labels = labels.to(device)


class CNNModel(nn.Module):
    def __init__(self, input_size, num_classes, dropout_rate=0.5):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.fc_input_size = self._calculate_fc_input_size(input_size)
        self.fc1 = nn.Linear(self.fc_input_size, 256)

        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=dropout_rate)

    def _calculate_fc_input_size(self, input_size):
        conv1_output_size = (input_size - 2) // 2
        conv2_output_size = (conv1_output_size - 2) // 2
        fc_input_size = 128 * conv2_output_size
        return fc_input_size

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, self.fc_input_size)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

criterion = nn.CrossEntropyLoss()

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

def train_cnn(model, train_loader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')


def test_cnn(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_predicted = []
    all_targets = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_predicted.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    accuracy = correct / total
    precision = precision_score(all_targets, all_predicted, average='weighted')
    recall = recall_score(all_targets, all_predicted, average='weighted')
    f1 = f1_score(all_targets, all_predicted, average='weighted')

    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    print(f'Precision: {precision * 100:.2f}%')
    print(f'Recall: {recall * 100:.2f}%')
    print(f'F1 Score: {f1 * 100:.2f}%')
    return accuracy, precision, recall, f1

num_experiments = 10
all_avg_accuracies = []
all_avg_precisions = []
all_avg_recalls = []
all_avg_f1_scores = []

for test_size in [0.1, 0.2, 0.3, 0.4, 0.5]:
    logging.info(f"Starting experiments for test_size: {test_size}")

    avg_accuracies = []
    avg_precisions = []
    avg_recalls = []
    avg_f1_scores = []
    X_test_tsne = []

    for experiment in range(num_experiments):
        X_train, X_test, y_train, y_test = train_test_split(combined_vectors, labels.cpu().numpy(), test_size=test_size,
                                                            random_state=42)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

        train_dataset = CustomDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        test_dataset = CustomDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        cnn_model = CNNModel(input_size=features.shape[1], num_classes=len(set(labels.cpu().numpy()))).to(device)
        optimizer = optim.Adam(cnn_model.parameters(), lr=0.001, weight_decay=1e-4)
        train_cnn(cnn_model, train_loader, optimizer, criterion, epochs=10)

        accuracy, precision, recall, f1 = test_cnn(cnn_model, test_loader)
        avg_accuracies.append(accuracy)
        avg_precisions.append(precision)
        avg_recalls.append(recall)
        avg_f1_scores.append(f1)
        logging.info(
            f"Test Size: {test_size}, Experiment: {experiment}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

    avg_accuracy = np.mean(avg_accuracies)
    avg_precision = np.mean(avg_precisions)
    avg_recall = np.mean(avg_recalls)
    avg_f1 = np.mean(avg_f1_scores)
    all_avg_accuracies.append(avg_accuracy)
    all_avg_precisions.append(avg_precision)
    all_avg_recalls.append(avg_recall)
    all_avg_f1_scores.append(avg_f1)
    logging.info(f"Average Test Accuracy for test_size {test_size}: {avg_accuracy}")
    logging.info(f"Average Precision for test_size {test_size}: {avg_precision}")
    logging.info(f"Average Recall for test_size {test_size}: {avg_recall}")
    logging.info(f"Average F1 Score for test_size {test_size}: {avg_f1}")

