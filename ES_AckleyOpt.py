import numpy as np
from math import exp, sqrt, cos, pi, e

def objective(x, y):
    return -20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2))) - exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y))) + e + 20

def evaluate_population(population):
    return np.array([objective(x, y) for x, y in population])

def sg_es(solution_dim, mean_=None, std_=None, gamma=50, muy=20, num_generations=100):
    if mean_ is None:
        mean_ = np.zeros(solution_dim)
    if std_ is None:
        std_ = np.ones(solution_dim)
    
    print("Initial mean:", mean_, " and standard deviation:", std_)
    for generation in range(num_generations):
        samples = np.random.normal(mean_, std_, size=(gamma, solution_dim))
        fitness_values = evaluate_population(samples)
        
        sorted_indices = np.argsort(fitness_values)
        sorted_samples = samples[sorted_indices]
        best_solution = sorted_samples[0]
        best_fitness = fitness_values[sorted_indices[0]]
        print(f"Generation {generation+1}: Best Solution {best_solution} - Best Fitness {best_fitness}")

        samples_for_update = sorted_samples[:muy]
        
        mean_ = np.mean(samples_for_update, axis=0)
        std_ = np.std(samples_for_update, axis=0)
    
    return best_solution, best_fitness

solution_dim = 2
num_generations = 1000
init_std = np.array([2, 2])

best_solution, best_fitness = sg_es(solution_dim,std_=init_std, num_generations=num_generations)
print("Optimal solution found at:", best_solution)
print("Optimal function value:", best_fitness)
