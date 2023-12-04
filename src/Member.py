import os
import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from Genes import DEFAULT_GENES

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')

NEUTRAL_FEATURES_TENSOR \
    = torch.load(os.path.join("..", "res", "AffectnetData", "neutral_features.pt")).to(device)
NEUTRAL_LABELS_TENSOR \
    = torch.load(os.path.join("..", "res", "AffectnetData", "neutral_labels.pt")).to(device)

HAPPY_FEATURES_TENSOR \
    = torch.load(os.path.join("..", "res", "AffectnetData", "happy_features.pt")).to(device)
HAPPY_LABELS_TENSOR \
    = torch.load(os.path.join("..", "res", "AffectnetData", "happy_labels.pt")).to(device)

SAD_FEATURES_TENSOR \
    = torch.load(os.path.join("..", "res", "AffectnetData", "sad_features.pt")).to(device)
SAD_LABELS_TENSOR \
    = torch.load(os.path.join("..", "res", "AffectnetData", "sad_labels.pt")).to(device)

SURPRISE_FEATURES_TENSOR \
    = torch.load(os.path.join("..", "res", "AffectnetData", "surprise_features.pt")).to(device)
SURPRISE_LABELS_TENSOR \
    = torch.load(os.path.join("..", "res", "AffectnetData", "surprise_labels.pt")).to(device)

FEAR_FEATURES_TENSOR \
    = torch.load(os.path.join("..", "res", "AffectnetData", "fear_features.pt")).to(device)
FEAR_LABELS_TENSOR \
    = torch.load(os.path.join("..", "res", "AffectnetData", "fear_labels.pt")).to(device)

DISGUST_FEATURES_TENSOR \
    = torch.load(os.path.join("..", "res", "AffectnetData", "disgust_features.pt")).to(device)
DISGUST_LABELS_TENSOR \
    = torch.load(os.path.join("..", "res", "AffectnetData", "disgust_labels.pt")).to(device)

ANGER_FEATURES_TENSOR \
    = torch.load(os.path.join("..", "res", "AffectnetData", "anger_features.pt")).to(device)
ANGER_LABELS_TENSOR \
    = torch.load(os.path.join("..", "res", "AffectnetData", "anger_labels.pt")).to(device)

TEST_SIZE = 0.2


def fitness_function(metrics):
    return sum(metrics.values()) / len(metrics.values())


class AffectnetDataset(Dataset):
    def __init__(self, X, y):
        self.data = X
        self.labels = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.data[idx]
        label = self.labels[idx]
        return features, label


class FacialExpressionModel(nn.Module):
    def __init__(self,
                 num_hidden_layers,
                 dropout_probabilities,
                 linear_layer_dimensions,
                 has_batch_norms,
                 hidden_layer_activations):
        super(FacialExpressionModel, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(512, linear_layer_dimensions[0]))
        self.layers.append(nn.ReLU())
        for i in range(num_hidden_layers):
            self.layers.append(nn.Dropout(dropout_probabilities[i]))
            self.layers.append(nn.Linear(linear_layer_dimensions[i], linear_layer_dimensions[i + 1]))
            if has_batch_norms[i]:
                self.layers.append(nn.BatchNorm1d(linear_layer_dimensions[i + 1]))
            self.layers.append(hidden_layer_activations[i])
        self.layers.append(nn.Linear(linear_layer_dimensions[num_hidden_layers], 7))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Member:
    def __init__(self, genes):
        self.genes = genes
        self.model = FacialExpressionModel(
            genes["num_hidden_layers"],
            genes["dropout_layer_probabilities"],
            genes["linear_layer_dimensions"],
            genes["has_batch_norm"],
            genes["hidden_layer_activations"]
        ).to(device)
        self.optimizer = genes["optimizer"](
            self.model.parameters(),
            lr=self.genes["learning_rate"],
            weight_decay=self.genes["weight_decay"]
        )
        self.metrics = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
        self.fitness = 0.0
        self.training_loss = []
        self.eval_loss = []
        pass

    def train(self):
        train_size = int((1 - TEST_SIZE) * len(FEATURES_TENSOR))
        train_X, test_X = FEATURES_TENSOR[:train_size], FEATURES_TENSOR[train_size:]
        train_y, test_y = LABELS_TENSOR[:train_size], LABELS_TENSOR[train_size:]

        train_dataset = AffectnetDataset(train_X, train_y)
        test_dataset = AffectnetDataset(test_X, test_y)
        train_loader = DataLoader(train_dataset, batch_size=self.genes["batch_size"], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.genes["batch_size"], shuffle=True)

        all_predicted = []
        all_labels = []
        for _ in tqdm(range(self.genes["num_epochs"])):
            running_loss = 0.0
            self.model.train()
            for features, labels in train_loader:
                self.optimizer.zero_grad()
                predicted = self.model(features)
                loss = self.genes["criteria"](predicted, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            self.training_loss.append(running_loss / len(train_loader))

            running_loss = 0.0
            self.model.eval()
            for features, labels in test_loader:
                predicted = self.model(features)
                loss = self.genes["criteria"](predicted, labels)
                running_loss += loss.item()
            self.eval_loss.append(running_loss / len(test_loader))

        self.model.eval()
        for features, labels in test_loader:
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
        pass

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
        pass

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
        pass


member = Member(DEFAULT_GENES)
member.train()
member.save(os.path.join("..", "res", "Saves", "test"), "member_test")
print(member.fitness)
