import numpy as np
from numpy.random import default_rng
# going to add numba or cython if the project becomes too complex

rng = default_rng()

# population initializer func
def pop_init(p_size):
	gene_pool = list("abcdefghijklmnopqrstuvwxyz")
	gene_ext = list(",.<>/-_?:;[]{}()|\\\'\"")

	# every layout has 35 keys total
	pop = np.empty((0, 35), 'U')

	for _ in range(p_size):

		current = np.array(np.concatenate((gene_pool, rng.choice(gene_ext, size=9, replace=False))), ndmin=2)

		rng.shuffle(current, axis=1)
		pop = np.concatenate((pop, current))

	return pop


# as the name suggests it will calculate fitness score for each chromosome
def calc_fitness(p):
	# key_bias = np.array([[4, 2, 2, 2.5, 3.5, 5, 2.5, 2, 2, 3.5, 4.5, 4.5], [1.5, 1, 1, 1, 2.5, 2.5, 1, 1, 1, 1.5, 3, 3], [3.5, 4, 3, 2.5, 2, 2.5, 2, 2.5, 3, 4, 3.5]])
	# key_distance = np.array([[1.032, 1.032, 1.032, 1.032, ], [], []])
	pass

# when used, will swap some genes
def mutation():
	pass

# main function to combine two chromosomes
def crossover():
	pass

# create next generation by combining fittest chromosomes
def next_iter():
	pass


# might change how i store these later
# gene_pool = np.array(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"])
# alphabet_ext = np.array([",", ".", "<", ">", "/", "?", ":", ";", "[", "]", "{", "}", "\'", "\"", "\\", "|"])


pop = pop_init(10)

print(pop)


