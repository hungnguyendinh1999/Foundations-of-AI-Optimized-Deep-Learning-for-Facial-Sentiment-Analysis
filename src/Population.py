import os
import random

import torch

from Genes import get_random_gene, get_crossover_gene, get_mutated_gene
from Member import Member


class Population:
    def __init__(self, population_size, survival_rate, mutation_rate, save_path):
        self.population_size = population_size
        self.survival_rate = survival_rate
        self.mutation_rate = mutation_rate
        self.save_path = save_path

        # population properties
        self.generation = 0
        self.genes = []
        self.fitness_scores = []

        # population members
        self.members = []
        self.best_genes = None
        self.best_member = None
        self.best_fitness = 0
        pass

    def start(self):
        while len(self.genes) < self.population_size:
            self.genes.append(get_random_gene())
            self.fitness_scores.append(0)
            self.members.append(None)
        print("initial population:", len(self.genes))
        pass

    def populate(self):
        for i, gene in enumerate(self.genes):
            if self.members[i] is None:
                self.members[i] = Member(gene)
                self.members[i].train()
                self.fitness_scores[i] = self.members[i].fitness
                if self.members[i].fitness > self.best_fitness:
                    self.best_genes = self.genes[i]
                    self.best_member = self.members[i]
                    self.best_fitness = self.members[i].fitness
                    self.best_member.save(os.path.join(self.save_path), "best_member")
            self.members[i].save(os.path.join(self.save_path, f"generation_{self.generation}"), f"member_{i}")
        pass

    def cull(self):
        zipped_sorted = sorted(zip(self.genes, self.fitness_scores, self.members), key=lambda x: x[1], reverse=True)
        self.genes, self.fitness_scores, self.members = map(list, zip(*zipped_sorted))
        num_survivors = int(self.survival_rate * len(self.genes))
        self.genes = self.genes[:num_survivors]
        self.fitness_scores = self.fitness_scores[:num_survivors]
        self.members = self.members[:num_survivors]
        print("post-culling population:", len(self.genes))
        pass

    def crossover(self):
        survivors = self.genes.copy()
        while len(self.genes) < self.population_size:
            parent1, parent2 = random.sample(survivors, 2)
            self.genes.append(get_crossover_gene(parent1, parent2))
            self.fitness_scores.append(0)
            self.members.append(None)
        print("post-crossover population:", len(self.genes))
        pass

    def mutate(self):
        mutation_count = 0
        for i, gene in enumerate(self.genes):
            if gene == self.best_genes:
                continue
            if random.random() < self.mutation_rate:
                self.genes[i] = get_mutated_gene(gene)
                self.fitness_scores[i] = 0
                self.members[i] = None
                mutation_count += 1
        print("mutations:", mutation_count)
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
        pass

    def load(self):
        if not os.path.exists(os.path.join(self.save_path, "population_properties.pt")):
            raise FileNotFoundError("Population properties file not found")
        properties = torch.load(os.path.join(self.save_path, "population_properties.pt"))
        self.generation = properties["generation"]
        self.genes = properties["genes"]

        member_properties = torch.load(os.path.join(self.save_path, "best_member_properties.pt"))
        self.best_genes = member_properties["genes"]
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
                self.fitness_scores[i] = 0
                self.members[i] = None
                break
        pass

    def run(self, num_generations, resume=False):
        if resume is True:
            self.load()
        else:
            self.start()
        for i in range(self.generation, num_generations):
            print(f"Generation {self.generation}")
            self.populate()
            self.cull()
            self.crossover()
            self.mutate()
            self.generation += 1
            self.save()
        pass


population = Population(
    64,
    0.5,
    1 / 16,
    os.path.join("..", "res", "fer2013Saves", "population")
)
population.run(16, resume=True)
