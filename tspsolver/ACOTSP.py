import numpy as np

from tspsolver.BaseTSPSolver import BaseTSPSolver

class ACOTSP(BaseTSPSolver):

    def __init__(self, 
                input_file_path: str = 'input.txt', 
                num_ants: int = 10, 
                num_iterations: int = 100, 
                alpha: float = 1.0, 
                beta: float = 5.0, 
                rho: float = 0.5,
                Q: float = 100,
                b: int = 5
            ) -> None:
        super().__init__()
        self.dmatrix = self.read_dmatrix(input_file_path)
        self.num_cities = self.dmatrix.shape[0]
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.b = b
        self.pheromone_matrix = self.init_pheromone_matrix()

    def init_pheromone_matrix(self):
        return np.random.random(size=self.dmatrix.shape)
    
    def calculate_tour_length(self, solution: np.array) -> float:
        total_cost = np.sum([self.dmatrix[solution[i]][solution[i+1]] for i in range(self.num_cities - 1)])
        total_cost += self.dmatrix[solution[-1]][solution[0]]
        return total_cost
    
    def create_ant_tours(self):
        ant_tours = []
        for ant in range(self.num_ants):
            remain = set(np.arange(self.num_cities))
            current_city = np.random.randint(0, self.num_cities)
            tour = [current_city]
            remain.remove(current_city)

            for _ in range(self.num_cities - 1):
                probabilities = self.calculate_probabilities(current_city, remain)
                next_city = np.random.choice(self.num_cities, 1, p=probabilities)[0]
                tour.append(next_city)
                remain.remove(next_city)
                current_city = next_city

            ant_tours.append((tour, self.calculate_tour_length(tour)))

        return ant_tours

    def calculate_probabilities(self, current_city: int, remain_cities: int):
        nuy_ = np.zeros(self.num_cities)
        for next_city in range(self.num_cities):
            if next_city in remain_cities:
                if (next_city != current_city) and (self.dmatrix[current_city][next_city] != 0):
                    nuy_[next_city] = 1 / self.dmatrix[current_city][next_city]

        nuy_ = nuy_ ** self.beta
        tau_ = self.pheromone_matrix[current_city] ** self.alpha
        proba = nuy_ * tau_
        return proba / np.sum(proba)

    def update_pheromones(self, ant_tours: np.ndarray):
        self.pheromone_matrix *= 1 - self.rho

        for tour, tour_length in ant_tours:
            for i in range(self.num_cities - 1):
                city1, city2 = tour[i], tour[i + 1]
                self.pheromone_matrix[city1][city2] += self.Q / tour_length
                self.pheromone_matrix[city2][city1] += self.Q / tour_length
    
    def postupdate_pheromones(self, best_tour: np.array, best_length: int):
        for i in range(self.num_cities - 1):
            city1, city2 = best_tour[i], best_tour[i + 1]
            self.pheromone_matrix[city1][city2] += self.Q / best_length
            self.pheromone_matrix[city2][city1] += self.Q / best_length      

    def run(self, verbose=False):
        ant_tours = self.create_ant_tours()
        self.update_pheromones(ant_tours)

        self.best_tour_, self.best_length_ = min(ant_tours, key=lambda x: x[1])
        self.postupdate_pheromones(self.best_tour_, self.best_length_)

        for iteration in range(self.num_iterations - 1):
            ant_tours = self.create_ant_tours()
            self.update_pheromones(ant_tours)

            best_tour, best_length = min(ant_tours, key=lambda x: x[1])
            if best_length < self.best_length_:
                self.best_length_ = best_length
                self.best_tour_ = best_tour

            self.postupdate_pheromones(best_tour, best_length)

            if verbose:
                print(f'[INFO] At iteration {iteration}, best tour length: {best_length}')

        return self.best_tour_, self.best_length_

