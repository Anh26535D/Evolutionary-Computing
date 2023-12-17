import random
import sympy as sp
import math

class Node:
    def __init__(self, value=None, left_node=None, right_node=None):
        self.value = value
        self.left_node = left_node
        self.right_node = right_node

    def to_string(self):
        if self.value in ExpressionTree.function_set:
            return f"({self.left_node.to_string()} {self.value} {self.right_node.to_string()})"
        else:
            return str(self.value)

class ExpressionTree:
    terminal_set = ["1", "X", "X**2", "X**3"]
    function_set = ["+", "-", "*", "/"]

    @staticmethod
    def random_tree(depth=1):
        if depth <= 1:
            value = random.choice(ExpressionTree.terminal_set)
            return Node(value)
        else:
            value = random.choice(ExpressionTree.function_set)
            left_subtree = ExpressionTree.random_tree(depth - 1)
            right_subtree = ExpressionTree.random_tree(depth - 1)
            return Node(value, left_subtree, right_subtree)
        
    @staticmethod
    def eval_tree(expression, var, val_of_var):
        expression = expression.replace(var, str(val_of_var))
        try:
            return eval(expression)
        except ZeroDivisionError as zero_error:
            return 100000
            
    @staticmethod
    def clone_tree(tree):
        if tree is None:
            return None
        return Node(tree.value, ExpressionTree.clone_tree(tree.left_node), ExpressionTree.clone_tree(tree.right_node))
    
    @staticmethod
    def select_random_node(tree):
        nodes = []

        def traverse(node):
            if node is not None:
                nodes.append(node)
                traverse(node.left_node)
                traverse(node.right_node)

        traverse(tree)
        return random.choice(nodes)

def load_ds(file_path):
    with open(file=file_path, mode="r") as f:
        ds = [[float(x) for x in line.split()] for line in f]
        return ds

dataset = load_ds("INPUT.txt")
num_samples = len(dataset)

class GA:
    sol_per_pop = 1000
    num_generations = 50
    mutation_rate = 0.01
    
    def __init__(self, sol_per_pop=None, num_generations=None, mutation_rate=None) -> None:
        if sol_per_pop is not None:
            self.sol_per_pop = sol_per_pop
        if num_generations is not None:
            self.num_generations = num_generations
        if mutation_rate is not None:
            self.mutation_rate = mutation_rate

    def init_population(self, depth):
        return [ExpressionTree.random_tree(depth=depth) for i in range(self.sol_per_pop)]

    def cal_fitness(self, solution):
        error = 0
        for val_x, y in dataset:
            y_hat = ExpressionTree.eval_tree(solution.to_string(), "X", val_of_var=val_x)
            error += (y - y_hat) ** 2
        return error / num_samples

    def mutate(self, solution):
        # Randomly select a node in the solution and mutate it
        random_node = ExpressionTree.select_random_node(solution)
        if random_node.value in ExpressionTree.terminal_set:
            random_node.value = random.choice(ExpressionTree.terminal_set)
        else:
            random_node.value = random.choice(ExpressionTree.function_set)

    def crossover(self, parent1, parent2):
        # Clone parents to create offspring
        offspring1 = ExpressionTree.clone_tree(parent1)
        offspring2 = ExpressionTree.clone_tree(parent2)

        # Select random nodes from each parent
        node_parent1 = ExpressionTree.select_random_node(offspring1)
        node_parent2 = ExpressionTree.select_random_node(offspring2)

        # Swap subtrees
        node_parent1.value, node_parent2.value = node_parent2.value, node_parent1.value
        node_parent1.left_node, node_parent2.left_node = node_parent2.left_node, node_parent1.left_node
        node_parent1.right_node, node_parent2.right_node = node_parent2.right_node, node_parent1.right_node

        return offspring1, offspring2

    def run(self):
        population = self.init_population(6)

        for generation in range(self.num_generations):
            population.sort(key=self.cal_fitness)
            best_solution = population[0]
            print("Best Solution at generation:", generation, "with fitness", self.cal_fitness(best_solution))
            print(best_solution.to_string())

            parents = population[:self.sol_per_pop // 2]

            # Create offspring through crossover and mutation
            offspring = [best_solution]  # Add the best solution from the previous generation

            while len(offspring) < self.sol_per_pop:
                parent1, parent2 = random.choices(parents, k=2)
                child1, child2 = self.crossover(parent1, parent2)
                offspring.extend([child1, child2])

            for i in range(self.sol_per_pop):
                if i > 0 and random.random() < self.mutation_rate:  # Skip mutation for the best solution
                    self.mutate(offspring[i])
            population = offspring

        best_solution = population[0]
        X = sp.symbols('X')
        sympy_expression = sp.sympify(best_solution.to_string())

        # Simplify the SymPy expression
        simplified_expression = sp.simplify(sympy_expression)

        print("Best Solution:")
        print(best_solution.to_string())
        print("Simplified best solution:")
        print(simplified_expression)

if __name__ == "__main__":
    GA().run()