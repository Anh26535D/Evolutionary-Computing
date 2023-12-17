import random
import math

sol_per_pop = 1000
num_generations = 100
mutation_rate = 0.01


def load_ds(file_path):
    with open(file=file_path, mode="r") as f:
        ds = [[float(x) for x in line.split()] for line in f]
        return ds


dataset = load_ds("INPUT.txt")
num_samples = len(dataset)


def init_population():
    population = []
    for _ in range(sol_per_pop):
        a = random.randint(-10, 10)
        b = random.randint(-10, 10)
        c = random.randint(-10, 10)
        population.append((a, b, c))
    return population


def cal_fitness(solution):
    loss = 0
    for X, y in dataset:
        loss += (y - (solution[0]*(X**2) + solution[1]*(X) + solution[0]))**2
    return math.sqrt(loss)/num_samples


def crossover(parent1, parent2, eta=2, epsilon=1e-6):
    child1, child2 = [], []
    for gene1, gene2 in zip(parent1, parent2):
        if random.random() < 0.5:
            if abs(gene1 - gene2) > epsilon:
                beta = 1.0 / (2.0 - (gene1 - gene2) /
                              (max(gene1, gene2) - min(gene1, gene2) + epsilon))
            else:
                beta = 1.0
            betaq = ((random.random() * (1.0 / beta + 1.0)) ** (1.0 / (eta + 1))
                     ) if beta > 1.0 else ((1.0 / (2.0 - beta) ** (1.0 / (eta + 1))))
            child_gene1 = 0.5 * (((1 + betaq) * gene1) + ((1 - betaq) * gene2))
            child_gene2 = 0.5 * (((1 - betaq) * gene1) + ((1 + betaq) * gene2))
        else:
            child_gene1, child_gene2 = gene1, gene2
        child1.append(child_gene1)
        child2.append(child_gene2)
    return tuple(child1), tuple(child2)


def mutate(solution):
    mutation_rate = 0.1
    mutated_solution = []
    for gene in solution:
        if random.random() < mutation_rate:
            mutation_value = random.gauss(mu=0, sigma=1)
            gene += mutation_value
        mutated_solution.append(gene)
    return tuple(mutated_solution)


def run_ga():
    population = init_population()

    for generation in range(num_generations):
        population.sort(key=cal_fitness)

        best_solution = population[0]
        print("Best Solution at generation:", generation,
              "with fitness", cal_fitness(best_solution))
        # print(best_solution)

        parents = population[:sol_per_pop // 2]
        offspring = [best_solution]
        while len(offspring) < sol_per_pop:
            parent1, parent2 = random.choices(parents, k=2)

            child1, child2 = crossover(parent1, parent2)
            offspring.extend([child1, child2])

        for i in range(sol_per_pop):
            if i > 0 and random.random() < mutation_rate:  # Skip mutation for the best solution
                mutate(offspring[i])
        population = offspring

    best_solution = population[0]
    print("Best Solution:")
    print(best_solution)
    print(len(population))


run_ga()
