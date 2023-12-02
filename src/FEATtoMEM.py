import os
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')


FEATURES_TENSOR = torch.load(os.path.join("..", "res", "AffectnetData", "features.pt")).to(device)
LABELS_TENSOR = torch.load(os.path.join("..", "res", "AffectnetData", "labels.pt")).to(device)
DEFAULT_GENES = {
    "learning_rate": 0.001,
    "num_epochs": 32,
    "batch_size": 64,
    "criteria": nn.CrossEntropyLoss(),
    "optimizer": optim.Adam,
    "num_hidden_layers": 5,
    "hidden_layer_dimensions": [256, 128, 64, 32, 16, 8],
    "layer_activations": [nn.ReLU() for _ in range(7)]
}
TEST_SIZE = 0.2


def fitness_function(metrics):
    return sum(metrics.values()) / len(metrics.values())


class AffectnetDataset(Dataset):
    def __init__(self, X, y, model):
        self.data = X
        self.labels = y
        self.model = model

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.data[idx]
        label = self.labels[idx]
        return features, label


class FacialExpressionModel(nn.Module):
    def __init__(self, num_hidden_layers, hidden_layer_dimensions, layer_activations):
        super(FacialExpressionModel, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_dimensions = hidden_layer_dimensions
        self.layer_activations = layer_activations

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(512, self.hidden_layer_dimensions[0]))
        for i in range(self.num_hidden_layers):
            self.layers.append(
                nn.Linear(self.hidden_layer_dimensions[i], self.hidden_layer_dimensions[i + 1]))
        self.layers.append(
            nn.Linear(self.hidden_layer_dimensions[self.num_hidden_layers], 7))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # Apply activation to all but last layer
                x = self.layer_activations[i](x)
        return x


class Member:
    def __init__(self, genes):
        self.genes = genes
        self.model = FacialExpressionModel(
            genes["num_hidden_layers"],
            genes["hidden_layer_dimensions"],
            genes["layer_activations"]
        ).to(device)
        self.metrics = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
        self.fitness = 0.0
        self.training_loss = []

    def train(self):
        train_size = int((1 - TEST_SIZE) * len(FEATURES_TENSOR))
        train_X, test_X = FEATURES_TENSOR[:train_size], FEATURES_TENSOR[train_size:]
        train_y, test_y = LABELS_TENSOR[:train_size], LABELS_TENSOR[train_size:]

        train_dataset = AffectnetDataset(train_X, train_y, self.model)
        test_dataset = AffectnetDataset(test_X, test_y, self.model)
        train_loader = DataLoader(train_dataset, batch_size=self.genes["batch_size"], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.genes["batch_size"], shuffle=True)

        optimizer = self.genes["optimizer"](self.model.parameters(), lr=self.genes["learning_rate"])

        all_predicted = []
        all_labels = []
        for epoch in tqdm(range(self.genes["num_epochs"])):
            running_loss = 0.0
            self.model.train()
            for i, (features, labels) in enumerate(train_loader, 0):
                optimizer.zero_grad()
                predicted = self.model(features)
                loss = self.genes["criteria"](predicted, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            self.training_loss.append(running_loss)

            if epoch + 1 == self.genes["num_epochs"]:
                self.model.eval()
                for i, (features, labels) in enumerate(test_loader, 0):
                    predicted = self.model(features)
                    predicted = torch.argmax(predicted, dim=1)
                    labels = torch.argmax(labels, dim=1)
                    all_predicted.extend(predicted.tolist())
                    all_labels.extend(labels.tolist())

        self.metrics["accuracy"] = accuracy_score(all_labels, all_predicted)
        self.metrics["precision"] = precision_score(all_labels, all_predicted, average="macro")
        self.metrics["recall"] = recall_score(all_labels, all_predicted, average="macro")
        self.metrics["f1"] = f1_score(all_labels, all_predicted, average="macro")

        self.fitness = fitness_function(self.metrics)

    def save(self, directory, prefix):
        if not os.path.exists(directory):
            os.makedirs(directory)

        torch.save(self.model.state_dict(), os.path.join(directory, prefix + "_model.pt"))

        properties = {
            "genes": self.genes,
            "metrics": self.metrics,
            "fitness": self.fitness,
            "training_loss": self.training_loss
        }

        torch.save(properties, os.path.join(directory, prefix + "_properties.pt"))

    def load(self, directory, prefix):
        if not os.path.exists(directory):
            raise FileNotFoundError("File does not exist")

        model_path = os.path.join(directory, prefix + "_model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError("File does not exist")
        self.model.load_state_dict(torch.load(model_path))

        properties_path = os.path.join(directory, prefix + "_properties.pt")
        if not os.path.exists(properties_path):
            raise FileNotFoundError("File does not exist")
        properties = torch.load(properties_path)
        self.genes = properties["genes"]
        self.metrics = properties["metrics"]
        self.fitness = properties["fitness"]
        self.training_loss = properties["training_loss"]


# member = Member(DEFAULT_GENES)
# member.train()
# member.save(os.path.join("..", "res", "Saves", "test"), "member_test")
# print(member.fitness)
