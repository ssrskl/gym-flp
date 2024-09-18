import math
import random
import gym
import gym_flp
import copy
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure
from gym_flp.util import FBSUtils


class GeneticAlgorithmFBS:
    def __init__(self, population_size, mutation_rate, crossover_rate, env, model):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.env = env
        self.model = model
        self.population = self.initialize_population()
        self.best_individual = None
        self.best_fitness = float("inf")
        model.learn(total_timesteps=1, reset_num_timesteps=False)

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            permutation, bay = FBSUtils.binary_solution_generator(
                self.env.area, self.env.n, self.env.fac_limit_aspect, self.env.L
            )
            population.append((permutation, bay))
        return population

    def evaluate_fitness(self, individual):
        permutation, bay = individual
        self.env.reset(layout=(permutation, bay))
        return self.env.Fitness

    def select_parents(self):
        fitnesses = [self.evaluate_fitness(ind) for ind in self.population]
        total_fitness = sum(fitnesses)
        probabilities = [f / total_fitness for f in fitnesses]
        population_index = len(self.population)
        parents = np.random.choice(population_index, size=2, p=probabilities)
        return self.population[parents[0]], self.population[parents[1]]

    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            point = random.randint(1, len(parent1[0]) - 1)
            child1 = (
                np.concatenate((parent1[0][:point], parent2[0][point:])),
                parent1[1],
            )
            child2 = (
                np.concatenate((parent2[0][:point], parent1[0][point:])),
                parent2[1],
            )
            return child1, child2
        return parent1, parent2

    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            permutation, bay = individual
            i, j = random.sample(range(len(permutation)), 2)
            permutation[i], permutation[j] = permutation[j], permutation[i]
        return individual

    def evolve(self):
        new_population = []
        for _ in range(self.population_size // 2):
            parent1, parent2 = self.select_parents()
            child1, child2 = self.crossover(parent1, parent2)
            new_population.append(self.mutate(child1))
            new_population.append(self.mutate(child2))
        self.population = new_population

    def run(self, generations):
        for generation in range(generations):
            self.evolve()
            for individual in self.population:
                fitness = self.evaluate_fitness(individual)
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_individual = individual
            print(f"Generation {generation}: Best Fitness = {self.best_fitness}")
        return self.best_individual, self.best_fitness


def main() -> None:
    instance = "Du62"
    env = gym.make("fbs-v0", instance=instance, mode="human")
    env.reset()

    model = DQN("MlpPolicy", env, verbose=1)

    ga = GeneticAlgorithmFBS(
        population_size=50, mutation_rate=0.1, crossover_rate=0.7, env=env, model=model
    )
    best_individual, best_fitness = ga.run(generations=100)

    print(f"Best permutation: {best_individual[0]}")
    print(f"Best fitness: {best_fitness}")
    env.reset(layout=best_individual)
    env.render()
    env.close()


if __name__ == "__main__":
    main()
