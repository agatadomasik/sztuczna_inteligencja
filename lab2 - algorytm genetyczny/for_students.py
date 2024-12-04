from itertools import compress
import random
import time
import matplotlib.pyplot as plt

from data import *

def initial_population(individual_size, population_size):
    return [[random.choice([True, False]) for _ in range(individual_size)] for _ in range(population_size)]

def fitness(items, knapsack_max_capacity, individual):
    total_weight = sum(compress(items['Weight'], individual))
    if total_weight > knapsack_max_capacity:
        return 0
    return sum(compress(items['Value'], individual))

def population_best(items, knapsack_max_capacity, population):
    best_individual = None
    best_individual_fitness = -1
    for individual in population:
        individual_fitness = fitness(items, knapsack_max_capacity, individual)
        if individual_fitness > best_individual_fitness:
            best_individual = individual
            best_individual_fitness = individual_fitness
    return best_individual, best_individual_fitness


def selection(population, items, knapsack_max_capacity, n_selection):
    fitnesses = [fitness(items, knapsack_max_capacity, individual) for individual in population]
    fitness_sum = sum(fitnesses)
    probabilities = [fit / fitness_sum for fit in fitnesses]
    selected_individuals = [];

    probablities_sum = [sum(probabilities[:i+1]) for i in range(len(probabilities))]
    for i in range(n_selection):
        rand = random.random()
        for j in range (len(probablities_sum)):
            if rand < probablities_sum[j]:
                selected_individuals.append(population[j])
                break


    # selected_indexes = random.choices(range(len(population)), weights=probabilities, k=n_selection)
    # selected_individuals = [population[i] for i in selected_indexes]

    return selected_individuals


def crossover(parent1, parent2):
    # x = random.randint(1, len(parent1) - 1)
    x = int(len(parent1)/2)
    child1 = parent1[:x] + parent2[x:]
    child2 = parent2[:x] + parent1[x:]
    return child1, child2


def mutation(individual, mutation_rate):
    mutated_individual = []
    for bit in individual:
        if random.random() < mutation_rate:
            mutated_individual.append(not bit)
        else:
            mutated_individual.append(bit)
    return mutated_individual

items, knapsack_max_capacity = get_big()
print(items)

population_size = 100
generations = 200
n_selection = 20
n_elite = 1

start_time = time.time()
best_solution = None
best_fitness = 0
population_history = []
best_history = []
population = initial_population(len(items), population_size)
for _ in range(generations):
    population_history.append(population)

    # TODO: implement genetic algorithm
    # Selection
    selected_population = selection(population, items, knapsack_max_capacity, n_selection)

    for _ in range (n_elite):
        elite = population_best(items, knapsack_max_capacity, population)[0]

    next_generation = [elite]

    # Crossover
    for _ in range(population_size - n_elite):
        parent1, parent2 = random.choices(selected_population, k=2)
        child1, child2 = crossover(parent1, parent2)
        next_generation.extend([mutation(child1, 0.05), mutation(child2, 0.05)])

    population = next_generation

    best_individual, best_individual_fitness = population_best(items, knapsack_max_capacity, population)
    if best_individual_fitness > best_fitness:
        best_solution = best_individual
        best_fitness = best_individual_fitness
    best_history.append(best_fitness)

end_time = time.time()
total_time = end_time - start_time
print('Best solution:', list(compress(items['Name'], best_solution)))
print('Best solution value:', best_fitness)
print('Time: ', total_time)

# plot generations
x = []
y = []
top_best = 10
for i, population in enumerate(population_history):
    plotted_individuals = min(len(population), top_best)
    x.extend([i] * plotted_individuals)
    population_fitnesses = [fitness(items, knapsack_max_capacity, individual) for individual in population]
    population_fitnesses.sort(reverse=True)
    y.extend(population_fitnesses[:plotted_individuals])
plt.scatter(x, y, marker='.')
plt.plot(best_history, 'r')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.show()