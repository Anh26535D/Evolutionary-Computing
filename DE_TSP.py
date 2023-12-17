import numpy as np

distance_matrix = [
    [0.0, 3.0, 4.0, 2.0, 7.0],
    [3.0, 0.0, 4.0, 6.0, 3.0],
    [4.0, 4.0, 0.0, 5.0, 8.0],
    [2.0, 6.0, 5.0, 0.0, 6.0],
    [7.0, 3.0, 8.0, 6.0, 0.0],
]

sol_per_pop = 10
num_generations = 100
mutation_rate = 0.8
crossover_rate = 0.7

class DifferentialEvolution():
    def __init__(self, sol_per_pop, num_generations, mutation_rate, crossover_rate, distance_matrix) -> None:
        self.sol_per_pop = sol_per_pop
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.distance_matrix = distance_matrix
        self.num_cities = len(distance_matrix)

    def init_population(self):
        population = np.empty((self.sol_per_pop, self.num_cities), dtype=int)
        for i in range(self.sol_per_pop):
            population[i] = np.random.permutation(self.num_cities)
        return population

    def fitness_func(self, solution):
        total_distance = 0
        for i in range(self.num_cities - 1):
            total_distance += self.distance_matrix[solution[i]][solution[i+1]]
        total_distance += self.distance_matrix[solution[-1]][solution[0]]
        return total_distance

    def mutation(self, target_vector, population):
        parents = population[np.random.choice(self.sol_per_pop, size=3, replace=False)]

        mutated_vector = parents[0] + self.mutation_rate * (parents[1] - parents[2])
        mutated_vector = np.clip(mutated_vector, 0, self.num_cities - 1).astype(int)

        crossover_mask = np.random.rand(self.num_cities) < self.crossover_rate
        child_vector = np.where(crossover_mask, mutated_vector, target_vector)

        unique_elements, counts = np.unique(child_vector, return_counts=True)
        duplicate_elements = unique_elements[counts > 1]

        for element in duplicate_elements:
            available_elements = np.setdiff1d(np.arange(self.num_cities), child_vector)
            child_vector[child_vector == element] = np.random.choice(available_elements)

        return child_vector

    def run(self, verbose=False):
        population = self.init_population()

        for generation in range(self.num_generations):
            best_solution = population[np.argmin(
                [self.fitness_func(solution) for solution in population])]
            if verbose:
                print(f"[INFO] Gen {generation}, best solution is {best_solution} has total distance: {self.fitness_func(best_solution)}")
        
            new_population = np.empty((self.sol_per_pop, self.num_cities), dtype=int)

            for i in range(self.sol_per_pop):
                target_vector = population[i]
                child_vector = self.mutation(target_vector, population)
                
                # Chọn cha mẹ và con tốt nhất giữa target_vector và child_vector
                if self.fitness_func(child_vector) < self.fitness_func(target_vector):
                    new_population[i] = child_vector
                else:
                    new_population[i] = target_vector

            population = new_population

        best_solution = population[np.argmin(
            [self.fitness_func(solution) for solution in population])]
        print(f"[DONE] Find best solution {best_solution} with fitness {self.fitness_func(best_solution)}")
        return best_solution, self.fitness_func(best_solution)

DE = DifferentialEvolution(
    sol_per_pop=sol_per_pop,
    num_generations=num_generations,
    mutation_rate=mutation_rate,
    crossover_rate=crossover_rate,
    distance_matrix=distance_matrix
)

best_solution, best_fitness = DE.run(verbose=True)
print("Best Solution:", best_solution)
print("Total Distance:", best_fitness)