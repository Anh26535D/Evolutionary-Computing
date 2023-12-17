import numpy as np
import scipy.linalg

# User input
N = 3 # number of objective variables dimension
num_generations = 100 # Number of iterations
xmean = np.random.random(N) # objective variables initial point
sigma = 0.3 # coordinate wise standard deviation (step size)

# Strategy parameter setting: Selection
lambda_ = int(4 + np.floor(3*np.log(N))) # population size, offspring number
mu = lambda_/2 # number of parents/points for recombination
mu = int(np.floor(mu))
weights = np.log(mu + 1/2) - np.log(np.arange(1, mu+1)) # weighted recombination
abs_w = np.abs(weights) ** 2
weights = abs_w / abs_w.sum(axis=0) # normalize recombination weights array
mueff = 1/np.sum(weights**2) # variance-effectiveness of sum w_i x_i

# Strategy parameter setting: Adaptation
cc = (4 + mueff/N)/(N + 4 + 2*mueff/N)  # time constant for cumulation for C
cs = (mueff + 2)/(N + mueff + 5);  # t-const for cumulation for sigma control
c1 = 2/((N + 1.3)**2 + mueff);    # learning rate for rank-one update of C
cmu = min(1 - c1, 2*(mueff - 2 + 1/mueff)/((N + 2)**2 + mueff));  # and for rank-mu update

# Initialize dynamic (internal) strategy parameters and constants
ps = np.zeros(N) # evolution paths for sigma
pc = np.zeros(N) # evolution paths for C
C = np.eye(N)

def rastrigin(x):
    """Rastrigin test objective function, shifted by 10. units away from origin"""
    x = np.copy(x)
    x -= 10.0
    if not np.isscalar(x[0]):
        N = len(x[0])
        return -np.array([10 * N + sum(xi**2 - 10 * np.cos(2 * np.pi * xi)) for xi in x])
    N = len(x)
    return -(10 * N + sum(x**2 - 10 * np.cos(2 * np.pi * x)))


def fitness(x):
    return -rastrigin(x)


def update_m(x, weights):
    return np.mean(x * weights, axis=0)


def update_p_sigma(ps, cs, C_sqrt_inv, displacement_m, mueff):
    discount_factor = 1 - cs
    complements_discount_factor = np.sqrt(1 - discount_factor**2)
    return discount_factor * ps + complements_discount_factor * np.sqrt(mueff) * (C_sqrt_inv @ displacement_m)


def update_p_c(pc, cc, mueff, displacement_m, norm_p_sigma, N):
    discount_factor = 1 - cc
    complements_discount_factor = np.sqrt(1 - discount_factor**2)
    indicator_func = 1 if norm_p_sigma <= 1.5 * np.sqrt(N) else 0
    return discount_factor * pc +  indicator_func * complements_discount_factor * np.sqrt(mueff) * displacement_m


def update_c(C, pc, x, sigma, m_prime, N, mu, norm_p_sigma):
    c_1 = 2/(N**2)
    c_muy = 1/(mu*N**2)
    if norm_p_sigma <= 1.5 * np.sqrt(N):
        c_s = 0
    else:
        c_s = c_1 * cc * (2 - cc)
    discount_factor = 1 - c_1 - c_muy + c_s

    rank_one_matrix = pc @ pc.T

    rank_min_muy_n_matrix = np.zeros_like(C)
    for xi in x:
        norm_x = (xi - m_prime)/sigma
        rank_min_muy_n_matrix += norm_x @ norm_x.T
    return discount_factor*C + c_1 * rank_one_matrix + c_muy * rank_min_muy_n_matrix


def update_sigma(sigma, cs, norm_p_sigma, N):
    return sigma * np.exp(cs * (norm_p_sigma/N - 1))


for _ in range(num_generations):
    x = np.random.multivariate_normal(xmean, C*sigma**2, lambda_)
    fx = [fitness(x_i) for x_i in x]
    sorted_idx = np.argsort(fx)
    sorted_x = x[sorted_idx]
    print(f"Best current solution at {_} is {sorted_x[0]} with fitness value is {rastrigin(sorted_x[0])}")

    selected_x = sorted_x[:mu]
    m_prime = xmean
    xmean = update_m(selected_x, weights)

    C_sqrt_inv = np.linalg.inv(scipy.linalg.sqrtm(C))
    displacement_m = (xmean - m_prime)/sigma
    ps = update_p_sigma(ps, cs, C_sqrt_inv, displacement_m, mueff)

    norm_p_sigma = np.linalg.norm(ps)
    pc = update_p_c(pc, cc, mueff, displacement_m, norm_p_sigma, N)

    C = update_c(C, pc, selected_x, sigma, m_prime, N, mu, norm_p_sigma)
    sigma = update_sigma(sigma, cs, norm_p_sigma, N)
