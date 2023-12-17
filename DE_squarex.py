import numpy as np

num_generations = 100

sol_per_pop = 100
sol_dim = 3
bounds = (-5, 5)

def fitness_func(x):
    return np.sum((x-1)**2)

def init_population(sol_per_pop, sol_dim, bounds):
    return np.random.uniform(bounds[0], bounds[1], size=(sol_per_pop, sol_dim))

def differential_evolution(fitness_func, cp=0.8, F=0.5, verbose=False):
    global num_generations, sol_dim, sol_per_pop, bounds

    population = init_population(sol_per_pop, sol_dim, bounds)

    for _ in range(num_generations):
        if verbose:
            best_index = np.argmin([fitness_func(individual) for individual in population])
            best_solution = population[best_index]
            print(f"Generation {_}: {fitness_func(best_solution)} with solution {best_solution}")

        for i in range(sol_per_pop):
            
            # Mutate
            indices = [idx for idx in range(sol_per_pop) if idx != i]
            a, b, c = np.random.choice(indices, 3, replace=False)
            vi = population[a] + F * (population[b] - population[c])

            # Crossover
            crossover_mask = np.random.rand(sol_dim) < cp
            vi[crossover_mask] = population[i, crossover_mask]
            oi = np.clip(vi, bounds[0], bounds[1])

            # Evaluate
            f_o = fitness_func(oi)
            f_i = fitness_func(population[i])

            # Select
            if f_o < f_i:
                population[i] = vi

    best_index = np.argmin([fitness_func(individual) for individual in population])
    best_solution = population[best_index]

    return best_solution, fitness_func(best_solution)

best_solution, minimum_value = differential_evolution(fitness_func, verbose=True)
print("====================================")
print("Minimum value found:", minimum_value)
print("Optimal parameter:", best_solution)
