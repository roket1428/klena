import corpus
import numpy as np

from numpy.random import default_rng
from nptyping import NDArray, Shape, Int, Unicode
# going to use numba or cython if the project becomes too complex

rng = default_rng()

# population initializer
def pop_init(p_size: int) -> NDArray[Shape["P_size, 35"], Unicode]:
	gene_pool = list("abcdefghijklmnopqrstuvwxyz")
	gene_ext = list(",.<>/?:;[]{}()|\\'\"")

	# every layout has 35 keys total
	pop = np.empty((0, 35), 'U')

	for _ in range(p_size):

		# gene_pool is 26 characters long and that leaves 9 for the symbols (extras)
		current = np.array(np.concatenate((gene_pool, rng.choice(gene_ext, size=9, replace=False))), ndmin=2)

		rng.shuffle(current, axis=1)
		pop = np.concatenate((pop, current))

	return pop


# as the name suggests it will calculate fitness score for each chromosome
def calc_fitness(pop: NDArray[Shape["P_size, 35"], Unicode], p_size: int) -> NDArray[Shape["*"], Int]:
	# key_bias = np.array([[4, 2, 2, 2.5, 3.5, 5, 2.5, 2, 2, 3.5, 4.5, 4.5], [1.5, 1, 1, 1, 2.5, 2.5, 1, 1, 1, 1.5, 3, 3], [3.5, 4, 3, 2.5, 2, 2.5, 2, 2.5, 3, 4, 3.5]])
	# key_distance = np.array([[1.032, 1.032, 1.032, 1.032, ], [], []])
	# calculate the distance for the word "hello world"
	"""
	scr_list = np.empty(0)

	for i in range(p_size):

		corpus = Corpus()
		current = np.array([pop[i,:12], pop[i,12:24], pop[i,24:]], dtype=object)
		for v in corpus:
			# TODO: implement this


		# scr_list = np.append(scr_list, )

	return scr_list """
	pass

# when used, will swap some genes
def mutate(chr: NDArray[Shape["35"], Unicode]) -> NDArray[Shape["35"], Unicode]:
	# array length - 1 so it doesn't overflow
	rnd_int = rng.integers(0, 34)
	chr[[rnd_int, rnd_int + 1]] = chr[[rnd_int + 1, rnd_int]]

	return chr

# main function to combine two chromosomes
def crossover(chr1, chr2):
	pass

# create next generation by combining fittest chromosomes
def next_iter():
	pass


# might change how i store these later
# gene_pool = np.array(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"])
# alphabet_ext = np.array([",", ".", "<", ">", "/", "?", ":", ";", "[", "]", "{", "}", "\'", "\"", "\\", "|"])

# tests
pop = pop_init(10)

print(pop, "\n")
print(pop[0])

test = np.array([pop[0,:12], pop[0,12:24], pop[0,24:]], dtype=object)
print(test[0])
print(test[1])
print(test[2])
#print(np.where(test=="h")[0])
#print(type(corpus.dictionary))



