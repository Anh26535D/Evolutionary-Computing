import numpy as np
from tspsolver.BaseTSPSolver import BaseTSPSolver

class PSOTSP(BaseTSPSolver):

    def __init__(self,
                 input_file_path: str = 'input.txt',
                 num_particles: int = 50,
                 num_iterations: int = 100,
                 inertia_weight: float = 0.5,
                 cognitive_coeff: float = 1.5,
                 social_coeff: float = 1.5,
            ) -> None:
        super().__init__()
        self.dmatrix = self.read_dmatrix(input_file_path)
        self.num_cities = self.dmatrix.shape[0]
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.inertia_weight = inertia_weight
        self.cognitive_coeff = cognitive_coeff
        self.social_coeff = social_coeff

    def init_particles(self):
        positions = np.random.permutation(self.num_cities)
        velocities = np.zeros_like(positions)
        return positions, velocities

    def evaluate_fitness(self, solution):
        total_cost = np.sum([self.dmatrix[solution[i]][solution[i+1]] for i in range(self.num_cities - 1)])
        total_cost += self.dmatrix[solution[-1]][solution[0]]
        return total_cost

    def update_velocity(self, positions, velocities, personal_best_positions, personal_best_fitness, global_best_positions):
        inertia_term = self.inertia_weight * velocities
        cognitive_term = self.cognitive_coeff * np.random.rand() * (personal_best_positions - positions)
        social_term = self.social_coeff * np.random.rand() * (global_best_positions - positions)
        new_velocities = inertia_term + cognitive_term + social_term
        return new_velocities

    def run(self, verbose=False):
        personal_best_positions = np.zeros((self.num_particles, self.num_cities), dtype=int)
        personal_best_fitness = np.zeros(self.num_particles)

        global_best_positions = None
        global_best_fitness = float('inf')

        for iteration in range(self.num_iterations):
            for particle in range(self.num_particles):
                if iteration == 0:
                    positions, velocities = self.init_particles()
                else:
                    velocities = self.update_velocity(positions, velocities, personal_best_positions[particle],
                                                      personal_best_fitness[particle], global_best_positions)
                    positions = np.argsort(np.argsort(positions + velocities))  # Ensure unique values in positions

                fitness = self.evaluate_fitness(positions)

                if fitness < personal_best_fitness[particle]:
                    personal_best_positions[particle] = positions.copy()
                    personal_best_fitness[particle] = fitness

                if fitness < global_best_fitness:
                    global_best_positions = positions.copy()
                    global_best_fitness = fitness

            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: Best Fitness = {global_best_fitness}")

        return global_best_positions, global_best_fitness
