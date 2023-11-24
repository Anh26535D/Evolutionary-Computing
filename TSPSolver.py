from tspsolver.ACOTSP import ACOTSP
from tspsolver.GATSP import GATSP


ALG_FLAG = 'ACO'
RUN_FLAG = True

if ALG_FLAG == 'GA':
    solver = GATSP()
elif ALG_FLAG == 'ACO':
    solver = ACOTSP()

if RUN_FLAG:
    best_sol, best_fitness = solver.run(verbose=True)
    print("Best solution:", best_sol)
    print("Total cost:", best_fitness)