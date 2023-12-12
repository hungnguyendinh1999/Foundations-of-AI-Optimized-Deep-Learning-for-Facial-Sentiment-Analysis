import random

import torch.nn as nn
import torch.optim as optim

from Random import truncated_normal, truncated_normal_int

GENE_SPACE = {
    "learning_rate": (0.0001, 0.1),
    "weight_decay": (0.0001, 0.1),
    "num_epochs": (8, 64),
    "batch_size": (8, 128),
    "criteria": [nn.CrossEntropyLoss(), nn.MSELoss(), nn.L1Loss(), nn.SmoothL1Loss()],
    "optimizer": [optim.Adam, optim.SGD, optim.RMSprop, optim.Adagrad, optim.AdamW],
    "num_hidden_layers": (1, 16),
    "linear_layer_dimensions": (8, 512),
    "dropout_layer_probabilities": (0, 0.5),
    "hidden_layer_activations": [nn.ReLU(), nn.Sigmoid(), nn.Tanh(), nn.LeakyReLU(), nn.ELU()]
}
DEFAULT_GENES = {
    "learning_rate": 0.001,
    "weight_decay": 0.001,
    "num_epochs": 32,
    "batch_size": 64,
    "criteria": nn.CrossEntropyLoss(),
    "optimizer": optim.Adam,
    "num_hidden_layers": 5,
    "dropout_layer_probabilities": [0, 0.25, 0, 0.25, 0],
    "linear_layer_dimensions": [256, 128, 64, 32, 16, 8],
    "has_batch_norm": [True, False, True, False, True],
    "hidden_layer_activations": [nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU()],
}


def get_random_learning_rate():
    return truncated_normal(
        DEFAULT_GENES["learning_rate"],
        GENE_SPACE["learning_rate"][0],
        GENE_SPACE["learning_rate"][1]
    )


def get_random_weight_decay():
    return truncated_normal(
        DEFAULT_GENES["weight_decay"],
        GENE_SPACE["weight_decay"][0],
        GENE_SPACE["weight_decay"][1]
    )


def get_random_num_epochs():
    return truncated_normal_int(
        DEFAULT_GENES["num_epochs"],
        GENE_SPACE["num_epochs"][0],
        GENE_SPACE["num_epochs"][1]
    )


def get_random_batch_size():
    return truncated_normal_int(
        DEFAULT_GENES["batch_size"],
        GENE_SPACE["batch_size"][0],
        GENE_SPACE["batch_size"][1]
    )


def get_random_criteria():
    return GENE_SPACE["criteria"][
        truncated_normal_int(0, 0, len(GENE_SPACE["criteria"]) - 1)
    ]


def get_random_optimizer():
    return GENE_SPACE["optimizer"][
        truncated_normal_int(0, 0, len(GENE_SPACE["optimizer"]) - 1)
    ]


def get_random_num_hidden_layers():
    return truncated_normal_int(
        DEFAULT_GENES["num_hidden_layers"],
        GENE_SPACE["num_hidden_layers"][0],
        GENE_SPACE["num_hidden_layers"][1]
    )


def get_random_dropout_layer_probabilities(num_hidden_layers):
    dropout_layer_probabilities = []
    for i in range(num_hidden_layers):
        default_layer_probability = 0.125
        if i < len(DEFAULT_GENES["dropout_layer_probabilities"]):
            default_layer_probability = DEFAULT_GENES["dropout_layer_probabilities"][i]
        dropout_layer_probabilities.append(truncated_normal(
            default_layer_probability,
            GENE_SPACE["dropout_layer_probabilities"][0],
            GENE_SPACE["dropout_layer_probabilities"][1]
        ))
    return dropout_layer_probabilities


def get_random_linear_layer_dimensions(num_hidden_layers):
    linear_layer_dimensions = []
    for i in range(num_hidden_layers + 1):
        default_layer_dimension = DEFAULT_GENES["linear_layer_dimensions"][-1]
        if i < len(DEFAULT_GENES["linear_layer_dimensions"]):
            default_layer_dimension = DEFAULT_GENES["linear_layer_dimensions"][i]
        linear_layer_dimensions.append(truncated_normal_int(
            default_layer_dimension,
            GENE_SPACE["linear_layer_dimensions"][0],
            GENE_SPACE["linear_layer_dimensions"][1]
        ))
    return linear_layer_dimensions


def get_random_has_batch_norm(num_hidden_layers):
    has_batch_norm = []
    for i in range(num_hidden_layers):
        has_batch_norm.append(random.random() < 0.25)
    return has_batch_norm


def get_random_hidden_layer_activations(num_hidden_layers):
    hidden_layer_activations = []
    for i in range(num_hidden_layers):
        hidden_layer_activations.append(
            GENE_SPACE["hidden_layer_activations"][truncated_normal_int(
                0,
                0,
                len(GENE_SPACE["hidden_layer_activations"]) - 1
            )])
    return hidden_layer_activations


def get_random_gene():
    random_gene = {
        "learning_rate": get_random_learning_rate(),
        "weight_decay": get_random_weight_decay(),
        "num_epochs": get_random_num_epochs(),
        "batch_size": get_random_batch_size(),
        "criteria": get_random_criteria(),
        "optimizer": get_random_optimizer(),
        "num_hidden_layers": get_random_num_hidden_layers(),
        "dropout_layer_probabilities": [],
        "linear_layer_dimensions": [],
        "has_batch_norm": [],
        "hidden_layer_activations": []
    }
    random_gene["dropout_layer_probabilities"] \
        = get_random_dropout_layer_probabilities(random_gene["num_hidden_layers"])
    random_gene["linear_layer_dimensions"] \
        = get_random_linear_layer_dimensions(random_gene["num_hidden_layers"])
    random_gene["has_batch_norm"] \
        = get_random_has_batch_norm(random_gene["num_hidden_layers"])
    random_gene["hidden_layer_activations"] \
        = get_random_hidden_layer_activations(random_gene["num_hidden_layers"])
    return random_gene


def get_crossover_learning_rate(gene1, gene2):
    return random.choice([gene1["learning_rate"], gene2["learning_rate"]])


def get_crossover_weight_decay(gene1, gene2):
    return random.choice([gene1["weight_decay"], gene2["weight_decay"]])


def get_crossover_num_epochs(gene1, gene2):
    return random.choice([gene1["num_epochs"], gene2["num_epochs"]])


def get_crossover_batch_size(gene1, gene2):
    return random.choice([gene1["batch_size"], gene2["batch_size"]])


def get_crossover_criteria(gene1, gene2):
    return random.choice([gene1["criteria"], gene2["criteria"]])


def get_crossover_optimizer(gene1, gene2):
    return random.choice([gene1["optimizer"], gene2["optimizer"]])


def get_crossover_num_hidden_layers(gene1, gene2):
    return random.choice([gene1["num_hidden_layers"], gene2["num_hidden_layers"]])


def get_crossover_dropout_layer_probabilities(gene1, gene2, crossover_num_hidden_layers):
    dropout_layer_probabilities = []
    for i in range(crossover_num_hidden_layers):
        if (i < len(gene1["dropout_layer_probabilities"])
                and i < len(gene2["dropout_layer_probabilities"])):
            dropout_layer_probabilities.append(
                random.choice([gene1["dropout_layer_probabilities"][i],
                               gene2["dropout_layer_probabilities"][i]])
            )
        elif i < len(gene1["dropout_layer_probabilities"]):
            dropout_layer_probabilities.append(gene1["dropout_layer_probabilities"][i])
        else:
            dropout_layer_probabilities.append(gene2["dropout_layer_probabilities"][i])
    return dropout_layer_probabilities


def get_crossover_linear_layer_dimensions(gene1, gene2, crossover_num_hidden_layers):
    linear_layer_dimensions = []
    for i in range(crossover_num_hidden_layers + 1):
        if (i < len(gene1["linear_layer_dimensions"])
                and i < len(gene2["linear_layer_dimensions"])):
            linear_layer_dimensions.append(
                random.choice(
                    [gene1["linear_layer_dimensions"][i], gene2["linear_layer_dimensions"][i]])
            )
        elif i < len(gene1["linear_layer_dimensions"]):
            linear_layer_dimensions.append(gene1["linear_layer_dimensions"][i])
        else:
            linear_layer_dimensions.append(gene2["linear_layer_dimensions"][i])
    return linear_layer_dimensions


def get_crossover_has_batch_norm(gene1, gene2, crossover_num_hidden_layers):
    has_batch_norm = []
    for i in range(crossover_num_hidden_layers):
        if i < len(gene1["has_batch_norm"]) and i < len(gene2["has_batch_norm"]):
            has_batch_norm.append(
                random.choice([gene1["has_batch_norm"][i], gene2["has_batch_norm"][i]])
            )
        elif i < len(gene1["has_batch_norm"]):
            has_batch_norm.append(gene1["has_batch_norm"][i])
        else:
            has_batch_norm.append(gene2["has_batch_norm"][i])
    return has_batch_norm


def get_crossover_hidden_layer_activations(gene1, gene2, crossover_num_hidden_layers):
    hidden_layer_activations = []
    for i in range(crossover_num_hidden_layers):
        if i < len(gene1["hidden_layer_activations"]) and i < len(
                gene2["hidden_layer_activations"]):
            hidden_layer_activations.append(
                random.choice(
                    [gene1["hidden_layer_activations"][i], gene2["hidden_layer_activations"][i]])
            )
        elif i < len(gene1["hidden_layer_activations"]):
            hidden_layer_activations.append(gene1["hidden_layer_activations"][i])
        else:
            hidden_layer_activations.append(gene2["hidden_layer_activations"][i])
    return hidden_layer_activations


def get_crossover_gene(gene1, gene2):
    crossover_gene = {
        "learning_rate": get_crossover_learning_rate(gene1, gene2),
        "weight_decay": get_crossover_weight_decay(gene1, gene2),
        "num_epochs": get_crossover_num_epochs(gene1, gene2),
        "batch_size": get_crossover_batch_size(gene1, gene2),
        "criteria": get_crossover_criteria(gene1, gene2),
        "optimizer": get_crossover_optimizer(gene1, gene2),
        "num_hidden_layers": get_crossover_num_hidden_layers(gene1, gene2),
        "dropout_layer_probabilities": [],
        "linear_layer_dimensions": [],
        "has_batch_norm": [],
        "hidden_layer_activations": []
    }
    crossover_gene["dropout_layer_probabilities"] \
        = get_crossover_dropout_layer_probabilities(gene1, gene2,
                                                    crossover_gene["num_hidden_layers"])
    crossover_gene["linear_layer_dimensions"] \
        = get_crossover_linear_layer_dimensions(gene1, gene2, crossover_gene["num_hidden_layers"])
    crossover_gene["has_batch_norm"] \
        = get_crossover_has_batch_norm(gene1, gene2, crossover_gene["num_hidden_layers"])
    crossover_gene["hidden_layer_activations"] \
        = get_crossover_hidden_layer_activations(gene1, gene2, crossover_gene["num_hidden_layers"])
    return crossover_gene


def get_mutated_learning_rate():
    return random.uniform(GENE_SPACE["learning_rate"][0], GENE_SPACE["learning_rate"][1])


def get_mutated_weight_decay():
    return random.uniform(GENE_SPACE["weight_decay"][0], GENE_SPACE["weight_decay"][1])


def get_mutated_num_epochs():
    return random.randint(GENE_SPACE["num_epochs"][0], GENE_SPACE["num_epochs"][1])


def get_mutated_batch_size():
    return random.randint(GENE_SPACE["batch_size"][0], GENE_SPACE["batch_size"][1])


def get_mutated_criteria():
    return random.choice(GENE_SPACE["criteria"])


def get_mutated_optimizer():
    return random.choice(GENE_SPACE["optimizer"])


def get_mutated_num_hidden_layers():
    return random.randint(GENE_SPACE["num_hidden_layers"][0], GENE_SPACE["num_hidden_layers"][1])


def get_mutated_dropout_layer_probabilities(mutated_num_hidden_layers):
    dropout_layer_probability = []
    for i in range(mutated_num_hidden_layers):
        dropout_layer_probability.append(
            random.uniform(GENE_SPACE["dropout_layer_probabilities"][0],
                           GENE_SPACE["dropout_layer_probabilities"][1])
        )
    return dropout_layer_probability


def get_mutated_linear_layer_dimensions(mutated_num_hidden_layers):
    hidden_layer_dimensions = []
    for i in range(mutated_num_hidden_layers + 1):
        hidden_layer_dimensions.append(
            random.randint(GENE_SPACE["linear_layer_dimensions"][0],
                           GENE_SPACE["linear_layer_dimensions"][1])
        )
    return hidden_layer_dimensions


def get_mutated_has_batch_norm(mutated_num_hidden_layers):
    has_batch_norm = []
    for i in range(mutated_num_hidden_layers):
        has_batch_norm.append(random.random() < 0.5)
    return has_batch_norm


def get_mutated_hidden_layer_activations(mutated_num_hidden_layers):
    hidden_layer_activations = []
    for i in range(mutated_num_hidden_layers):
        hidden_layer_activations.append(
            random.choice(GENE_SPACE["hidden_layer_activations"])
        )
    return hidden_layer_activations


def get_mutated_gene(gene):
    mutated_gene = gene.copy()
    trait = random.choice(list(gene.keys()))
    if trait == "learning_rate":
        mutated_gene[trait] = get_mutated_learning_rate()
        return mutated_gene
    if trait == "weight_decay":
        mutated_gene[trait] = get_mutated_weight_decay()
        return mutated_gene
    if trait == "num_epochs":
        mutated_gene[trait] = get_mutated_num_epochs()
        return mutated_gene
    if trait == "batch_size":
        mutated_gene[trait] = get_mutated_batch_size()
        return mutated_gene
    if trait == "criteria":
        mutated_gene[trait] = get_mutated_criteria()
        return mutated_gene
    if trait == "optimizer":
        mutated_gene[trait] = get_mutated_optimizer()
        return mutated_gene
    if trait == "num_hidden_layers":
        mutated_gene[trait] = get_mutated_num_hidden_layers()
        mutated_gene["dropout_layer_probabilities"] \
            = get_mutated_dropout_layer_probabilities(mutated_gene["num_hidden_layers"])
        mutated_gene["linear_layer_dimensions"] \
            = get_mutated_linear_layer_dimensions(mutated_gene["num_hidden_layers"])
        mutated_gene["has_batch_norm"] \
            = get_mutated_has_batch_norm(mutated_gene["num_hidden_layers"])
        mutated_gene["hidden_layer_activations"] \
            = get_mutated_hidden_layer_activations(mutated_gene["num_hidden_layers"])
        return mutated_gene
    if trait == "dropout_layer_probabilities":
        mutated_gene[trait] = get_mutated_dropout_layer_probabilities(len(mutated_gene[trait]))
        return mutated_gene
    if trait == "linear_layer_dimensions":
        mutated_gene[trait] = get_mutated_linear_layer_dimensions(mutated_gene["num_hidden_layers"])
        return mutated_gene
    if trait == "has_batch_norm":
        mutated_gene[trait] = get_mutated_has_batch_norm(mutated_gene["num_hidden_layers"])
        return mutated_gene
    if trait == "hidden_layer_activations":
        mutated_gene[trait] = get_mutated_hidden_layer_activations(
            mutated_gene["num_hidden_layers"])
        return mutated_gene
    return mutated_gene
