import os
import random
import torch
import torch.nn as nn
import torch.optim as optim

from scipy.stats import truncnorm
from FEATtoMEM import Member

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')

# FEATURES_TENSOR = torch.load(os.path.join("..", "res", "AffectnetData", "features.pt")).to(device)
# LABELS_TENSOR = torch.load(os.path.join("..", "res", "AffectnetData", "labels.pt")).to(device)
GENE_SPACE = {
    "learning_rate": (0.0001, 0.1),
    "num_epochs": (8, 64),
    "batch_size": (8, 128),
    "criteria": [nn.CrossEntropyLoss(), nn.MSELoss(), nn.L1Loss(), nn.BCEWithLogitsLoss()],
    "optimizer": [optim.Adam, optim.SGD, optim.RMSprop, optim.Adagrad, optim.AdamW],
    "num_hidden_layers": (1, 16),
    "hidden_layer_dimensions": (8, 512),
    "layer_activations": [nn.ReLU(), nn.Sigmoid(), nn.Tanh(), nn.LeakyReLU(), nn.ELU()]
}
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


def truncated_normal(center, low, high, size=1):
    sigma = (high - low) / 6
    a, b = (low - center) / sigma, (high - center) / sigma
    return truncnorm.rvs(a, b, loc=center, scale=sigma, size=size)[0]


def truncated_normal_int(center, low, high, size=1):
    return int(truncated_normal(center, low, high, size))


def get_random_gene():
    gene = {
        "learning_rate": truncated_normal(
            DEFAULT_GENES["learning_rate"],
            GENE_SPACE["learning_rate"][0],
            GENE_SPACE["learning_rate"][1]
        ),
        "num_epochs": truncated_normal_int(
            DEFAULT_GENES["num_epochs"],
            GENE_SPACE["num_epochs"][0],
            GENE_SPACE["num_epochs"][1]
        ),
        "batch_size": truncated_normal_int(
            DEFAULT_GENES["batch_size"],
            GENE_SPACE["batch_size"][0],
            GENE_SPACE["batch_size"][1]
        ),
        "criteria": random.choice(GENE_SPACE["criteria"]),
        "optimizer": random.choice(GENE_SPACE["optimizer"]),
        "num_hidden_layers": truncated_normal_int(
            DEFAULT_GENES["num_hidden_layers"],
            GENE_SPACE["num_hidden_layers"][0],
            GENE_SPACE["num_hidden_layers"][1]
        ),
        "hidden_layer_dimensions": [],
        "layer_activations": []
    }
    for _ in range(gene["num_hidden_layers"] + 1):
        gene["hidden_layer_dimensions"].append(
            truncated_normal_int(
                DEFAULT_GENES["hidden_layer_dimensions"][0],
                GENE_SPACE["hidden_layer_dimensions"][0],
                GENE_SPACE["hidden_layer_dimensions"][1]
            )
        )
    for _ in range(gene["num_hidden_layers"] + 2):
        gene["layer_activations"].append(random.choice(GENE_SPACE["layer_activations"]))
    return gene


def get_crossover_gene(gene1, gene2):
    crossover_gene = {
        "learning_rate": random.choice([gene1["learning_rate"], gene2["learning_rate"]]),
        "num_epochs": random.choice([gene1["num_epochs"], gene2["num_epochs"]]),
        "batch_size": random.choice([gene1["batch_size"], gene2["batch_size"]]),
        "criteria": random.choice([gene1["criteria"], gene2["criteria"]]),
        "optimizer": random.choice([gene1["optimizer"], gene2["optimizer"]]),
        "num_hidden_layers": random.choice([gene1["num_hidden_layers"], gene2["num_hidden_layers"]]),
        "hidden_layer_dimensions": [],
        "layer_activations": []
    }
    for i in range(crossover_gene["num_hidden_layers"] + 1):
        if i < len(gene1["hidden_layer_dimensions"]) and i < len(gene2["hidden_layer_dimensions"]):
            crossover_gene["hidden_layer_dimensions"].append(
                random.choice([gene1["hidden_layer_dimensions"][i], gene2["hidden_layer_dimensions"][i]])
            )
        elif i < len(gene1["hidden_layer_dimensions"]):
            crossover_gene["hidden_layer_dimensions"].append(gene1["hidden_layer_dimensions"][i])
        else:
            crossover_gene["hidden_layer_dimensions"].append(gene2["hidden_layer_dimensions"][i])
    for i in range(crossover_gene["num_hidden_layers"] + 2):
        if i < len(gene1["layer_activations"]) and i < len(gene2["layer_activations"]):
            crossover_gene["layer_activations"].append(
                random.choice([gene1["layer_activations"][i], gene2["layer_activations"][i]])
            )
        elif i < len(gene1["layer_activations"]):
            crossover_gene["layer_activations"].append(gene1["layer_activations"][i])
        else:
            crossover_gene["layer_activations"].append(gene2["layer_activations"][i])
    return crossover_gene


def get_mutated_gene(gene):
    mutated_gene = gene.copy()
    trait = random.choice(list(gene.keys()))
    if trait == "learning_rate":
        mutated_gene[trait] = random.uniform(GENE_SPACE[trait][0], GENE_SPACE[trait][1])
    elif trait == "num_epochs" or trait == "batch_size":
        mutated_gene[trait] = random.randint(GENE_SPACE[trait][0], GENE_SPACE[trait][1])
    elif trait == "num_hidden_layers":
        new_num_hidden_layers = random.randint(GENE_SPACE[trait][0], GENE_SPACE[trait][1])
        while new_num_hidden_layers == mutated_gene[trait]:
            new_num_hidden_layers = random.randint(GENE_SPACE[trait][0], GENE_SPACE[trait][1])
        mutated_gene[trait] = new_num_hidden_layers
        while mutated_gene[trait] + 1 > len(mutated_gene["hidden_layer_dimensions"]):
            mutated_gene["hidden_layer_dimensions"].append(
                random.randint(
                    GENE_SPACE["hidden_layer_dimensions"][0],
                    GENE_SPACE["hidden_layer_dimensions"][1]
                )
            )
        while mutated_gene[trait] + 1 < len(mutated_gene["hidden_layer_dimensions"]):
            mutated_gene["hidden_layer_dimensions"].pop()
        while mutated_gene[trait] + 2 > len(mutated_gene["layer_activations"]):
            mutated_gene["layer_activations"].append(random.choice(GENE_SPACE["layer_activations"]))
        while mutated_gene[trait] + 2 < len(mutated_gene["layer_activations"]):
            mutated_gene["layer_activations"].pop()
    elif trait == "hidden_layer_dimensions":
        i = random.randint(0, len(mutated_gene[trait]) - 1)
        mutated_gene[trait][i] = random.randint(
            GENE_SPACE["hidden_layer_dimensions"][0],
            GENE_SPACE["hidden_layer_dimensions"][1]
        )
    elif trait == "layer_activations":
        i = random.randint(0, len(mutated_gene[trait]) - 1)
        mutated_gene[trait][i] = random.choice(GENE_SPACE["layer_activations"])
    else: # trait == "criteria" or trait == "optimizer"
        mutated_gene[trait] = random.choice(GENE_SPACE[trait])
    return mutated_gene

class Population:
    def __init__(self, population_size, survival_rate, mutation_rate, crossover_rate, save_path, resume=False):
        self.population_size = population_size
        self.survival_rate = survival_rate
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.save_path = save_path

        self.resume = resume

        self.generation = 0
        self.genes = []
        self.fitness_scores = []

        self.members = []
        self.best_genes = None
        self.best_member = None
        self.best_fitness = 0

    def populate(self):
        while len(self.genes) < self.population_size:
            self.genes.append(get_random_gene())
        while len(self.members) < len(self.genes):
            self.fitness_scores.append(0)
            self.members.append(None)
        for i, gene in enumerate(self.genes):
            if self.members[i] is None:
                self.members[i] = Member(gene)
                self.members[i].train()
                self.fitness_scores[i] = self.members[i].fitness
                if self.members[i].fitness > self.best_fitness:
                    self.best_genes = self.genes[i]
                    self.best_member = self.members[i]
                    self.best_fitness = self.members[i].fitness
            self.members[i].save(os.path.join(self.save_path, f"generation_{self.generation}"), f"member_{i}")
        pass

    def cull(self):
        fitness_scores_sorted = sorted(self.fitness_scores, reverse=True)
        survival_threshold = fitness_scores_sorted[int(self.survival_rate * len(self.genes))]
        for i, member in enumerate(self.members):
            if member.fitness < survival_threshold:
                self.genes[i] = None
                self.members[i] = None
                self.fitness_scores[i] = None
        self.genes = [gene for gene in self.genes if gene is not None]
        self.members = [member for member in self.members if member is not None]
        self.fitness_scores = [score for score in self.fitness_scores if score is not None]
        pass

    def crossover(self):
        num_parents = int(self.crossover_rate * len(self.genes))
        if num_parents % 2 == 1:
            num_parents -= 1
        parents = random.sample(self.genes, num_parents)
        random.shuffle(parents)
        for i in range(0, len(parents), 2):
            self.genes.append(get_crossover_gene(parents[i], parents[i + 1]))
        pass

    def mutate(self):
        for i, gene in enumerate(self.genes):
            if random.random() < self.mutation_rate:
                self.genes[i] = get_mutated_gene(gene)
        pass

    def save(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        properties = {
            "generation": self.generation,
            "genes": self.genes,
            "best_genes": self.best_genes,
        }

        torch.save(properties, os.path.join(self.save_path, "population_properties.pt"))
        self.best_member.save(os.path.join(self.save_path), "best_member")

    def load(self):
        if not os.path.exists(os.path.join(self.save_path, "population_properties.pt")):
            raise FileNotFoundError("Population properties file not found")

        properties = torch.load(os.path.join(self.save_path, "population_properties.pt"))
        self.generation = properties["generation"]
        self.genes = properties["genes"]
        self.best_genes = properties["best_genes"]
        self.best_member = Member(self.best_genes)
        self.best_member.load(os.path.join(self.save_path), "best_member")
        self.best_fitness = self.best_member.fitness

        while len(self.members) < len(self.genes):
            self.members.append(None)
            self.fitness_scores.append(0)

        for i, gene in enumerate(self.genes):
            try:
                self.members[i] = Member(gene)
                self.members[i].load(os.path.join(self.save_path, f"generation_{self.generation}"), f"member_{i}")
                self.fitness_scores[i] = self.members[i].fitness
            except FileNotFoundError as e:
                break
        pass

    def run(self, num_generations):
        if self.resume is True:
            self.load()
        for i in range(self.generation, num_generations):
            print(f"Generation {self.generation}")
            self.populate()
            self.cull()
            self.crossover()
            self.mutate()
            self.generation += 1
            self.save()
        pass


population = Population(64, 0.666, 0.01, 1, os.path.join("..", "res", "Saves", "population"), resume=False)
population.run(32)