from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch_geometric.utils import add_self_loops, train_test_split_edges
from torch_geometric.nn import SAGEConv
import torch
from torch_geometric.data import Data
import pandas as pd
import networkx as nx
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import torch.optim as optim
import logging

from sklearn.model_selection import train_test_split

yuzhi = 0.8
alpha = 0.3

logging.basicConfig(filename=f'yuzhi_{yuzhi}_GraphSage_experiment.log_{alpha}', level=logging.INFO)

test_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]

combined_vectors = np.loadtxt("tfidf_doc2vec_features.csv", delimiter=",")
labels = torch.tensor(np.loadtxt("labels.csv", delimiter=","), dtype=torch.long)
num_features = combined_vectors.shape[1]
num_classes = len(set(labels.numpy()))
num_experiments = 10
num_epochs = 500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Graphsage(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Graphsage, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

def hybrid_sample_neighbors(edge_index, num_nodes, num_samples=10, similarity_matrix=None):
    row, col = edge_index
    sampled_neighbors = []
    for node in range(num_nodes):
        a = torch.rand(1).item()
        if a < alpha:
            sampled_neighbors.append(weighted_sample_neighbors(node, row, col, num_samples, similarity_matrix))
        else:
            sampled_neighbors.append(random_sample_neighbors(node, row, col, num_samples))
    sampled_neighbors = torch.cat(sampled_neighbors, dim=0)
    return sampled_neighbors

def random_sample_neighbors(node, row, col, num_samples):
    neighbors = col[row == node]
    if len(neighbors) <= num_samples:
        return neighbors
    else:
        sample_idx = torch.randint(0, len(neighbors), (num_samples,), dtype=torch.long)
        return neighbors[sample_idx]

def weighted_sample_neighbors(node, row, col, num_samples, similarity_matrix):
    neighbors = col[row == node]
    similarities = similarity_matrix[node, neighbors]
    if len(neighbors) <= num_samples:
        return neighbors
    else:
        top_k_neighbors = torch.topk(similarities, k=num_samples)[1]
        return neighbors[top_k_neighbors]

def train(model, data, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(data.x, data.train_pos_edge_index)
        loss = F.cross_entropy(output, data.y)
        l2_regularization = torch.tensor(0., requires_grad=True)
        for param in model.parameters():
            l2_regularization = l2_regularization + torch.sum(param ** 2)
        total_loss = loss + 5e-4 * l2_regularization
        total_loss.backward()
        optimizer.step()

def test(model, data):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.test_pos_edge_index)
        predicted_labels = logits.argmax(dim=1)
        accuracy = accuracy_score(data.y.cpu().numpy(), predicted_labels.cpu().numpy())
        precision = precision_score(data.y.cpu().numpy(), predicted_labels.cpu().numpy(), average='macro')
        recall = recall_score(data.y.cpu().numpy(), predicted_labels.cpu().numpy(), average='macro')
        f1 = f1_score(data.y.cpu().numpy(), predicted_labels.cpu().numpy(), average='macro')
        print(f'Test Accuracy: {accuracy * 100:.2f}%')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')

# 运行实验
for test_ratio in test_ratios:
    all_accuracies = []
    all_precisions = []
    all_recalls = []
    all_f1s = []
    all_embeddings = []
    logging.info(f"yuzhi_{yuzhi}_Running experiments with test ratio: {test_ratio}")
    for exp in range(num_experiments):
        x_train, x_test, y_train, y_test = train_test_split(combined_vectors, labels.cpu().numpy(), test_size=test_ratio, random_state=42)
        G = nx.Graph()
        node_labels = [label.item() for label in y_train]
        for i in range(len(y_train)):
            G.add_node(i, label=node_labels[i])
        similarities = np.array(cosine_similarity(x_train))
        for i in range(len(y_train)):
            for j in range(i + 1, len(y_train)):
                similarity = similarities[i][j]
                if similarity > yuzhi:
                    G.add_edge(i, j, similarity=similarity)
        edge_index = torch.tensor(np.array(G.edges()).T, dtype=torch.long).to(device)
        x_train = torch.tensor(x_train, dtype=torch.float).to(device)
        y_train = torch.tensor(y_train, dtype=torch.long).to(device)
        data = Data(x=x_train, y=y_train, edge_index=edge_index).to(device)
        data = train_test_split_edges(data, test_ratio=test_ratio)
        model = Graphsage(in_channels=num_features, hidden_channels=64, out_channels=num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        train(model, data, optimizer, num_epochs)
        test(model, data)

        with torch.no_grad():
            logits = model(data.x, data.test_pos_edge_index)
            predicted_labels = logits.argmax(dim=1)
            accuracy = accuracy_score(data.y.cpu().numpy(), predicted_labels.cpu().numpy())
            precision = precision_score(data.y.cpu().numpy(), predicted_labels.cpu().numpy(), average='macro')
            recall = recall_score(data.y.cpu().numpy(), predicted_labels.cpu().numpy(), average='macro')
            f1 = f1_score(data.y.cpu().numpy(), predicted_labels.cpu().numpy(), average='macro')

            all_accuracies.append(accuracy)
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_f1s.append(f1)
        embeddings = model(data.x, data.test_pos_edge_index)
        all_embeddings.append(embeddings.cpu().detach().numpy())

    average_accuracy = np.mean(all_accuracies)
    average_precision = np.mean(all_precisions)
    average_recall = np.mean(all_recalls)
    average_f1 = np.mean(all_f1s)

    logging.info(f"\nyuzhi_{yuzhi}_Average Test Accuracy over {num_experiments} experiments: {average_accuracy * 100:.2f}%")
    logging.info(f"yuzhi_{yuzhi}_Average Precision over {num_experiments} experiments: {average_precision:.4f}")
    logging.info(f"yuzhi_{yuzhi}_Average Recall over {num_experiments} experiments: {average_recall:.4f}")
    logging.info(f"yuzhi_{yuzhi}_Average F1 Score over {num_experiments} experiments: {average_f1:.4f}")
