import numpy as np

from tspsolver.BaseTSPSolver import BaseTSPSolver



class GATSP(BaseTSPSolver):

    def __init__(self, 
                input_file_path: str = 'input.txt', 
                num_generations: int = 100,
                sol_per_pop: int = 10, 
                mutation_rate: float = 0.01,
                population: np.ndarray = None
            ) -> None:
        super().__init__()
        self.sol_per_pop = sol_per_pop
        self.dmatrix = self.read_dmatrix(input_file_path)
        self.num_cities = self.dmatrix.shape[0]
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate

        if population is None:
            self.population = self.init_population()
        elif (population.shape[0] != sol_per_pop) or (population.shape[1] != self.num_cities):
            raise 'Population should match the shape of (sol_per_pop, num_cities)'
        else:
            self.population = population

    def init_population(self) -> np.ndarray:
        population = np.empty(shape=(self.sol_per_pop, self.num_cities))
        for i in range(self.sol_per_pop):
            population[i] = np.random.permutation(np.arange(self.num_cities))
        return population.astype('int')

    def cal_fitness(self, solution: np.array) -> float:
        total_cost = np.sum([self.dmatrix[solution[i]][solution[i+1]] for i in range(self.num_cities - 1)])
        total_cost += self.dmatrix[solution[-1]][solution[0]]
        return total_cost

    def crossover(self, parent_1: np.array, parent_2: np.array):
        crossover_point = np.random.randint(1, self.num_cities - 1)
        child = np.zeros_like(parent_1)
        child[:crossover_point] = parent_1[:crossover_point]
        for city in parent_2:
            if city not in child:
                child[crossover_point] = city
                crossover_point += 1
        return child

    def mutate(self, solution: np.array):
        if np.random.rand() < self.mutation_rate:
            idx1, idx2 = np.random.choice(self.num_cities, 2, replace=False)
            solution[idx1], solution[idx2] =  solution[idx2], solution[idx1]
        return solution

    def run(self, verbose=False):
        for generation in range(self.num_generations):
            fitness_pop = np.array([self.cal_fitness(solution) for solution in self.population])
            proba_fitness_pop = fitness_pop / np.sum(fitness_pop)
            sorted_indices = np.argsort(fitness_pop)
            sorted_population = self.population[sorted_indices]
            current_best_sol = sorted_population[0]
            if verbose:
                print(f'[INFO] Best total distance at generation {generation}: {self.cal_fitness(current_best_sol)}')

            new_population = np.empty_like(self.population)
            new_population[0] = current_best_sol
            for i in range(1, self.sol_per_pop):
                idx1, idx2 = np.random.choice(self.sol_per_pop, 2, p=proba_fitness_pop)
                new_population[i] = self.mutate(self.crossover(self.population[idx1], self.population[idx2]))

            self.population = new_population

        best_solution = self.population[np.argmin([self.cal_fitness(solution) for solution in self.population])]
        return best_solution, self.cal_fitness(best_solution)