import pathmap

import numpy as np

from numpy.random import default_rng
from nptyping import NDArray, Shape, Int, Unicode
from smart_open import open

class Corpus:
	def __iter__(self):
		for line in open("corpus/corpus_minimal.txt"):
			yield line.lower().split()

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
def calc_fitness(pop: NDArray[Shape["P_size, 35"], Unicode], p_size: int) -> dict:
	# key_bias = np.array([[4, 2, 2, 2.5, 3.5, 5, 2.5, 2, 2, 3.5, 4.5, 4.5], [1.5, 1, 1, 1, 2.5, 2.5, 1, 1, 1, 1.5, 3, 3], [3.5, 4, 3, 2.5, 2, 2.5, 2, 2.5, 3, 4, 3.5]])
	scr_list = {}

	for p in range(p_size):

		print("pop:", p)
		corpus = Corpus()
		chr = np.stack((pop[p,:12], pop[p,12:24], np.concatenate((pop[p,24:], np.array([np.nan])))))
		path_map = pathmap.mapgen(chr)

		distance = 0
		last_reg = -1
		for ind, doc in enumerate(corpus):
			print("corpus: ", ind)
			print("total dist:", distance)
			for w in doc:
				for c in w:
					try:
						cur_idy = np.where(chr==c)[0][0]
						cur_idx = np.where(chr==c)[1][0]
					except IndexError:
						continue

					if cur_idy == 2 and cur_idx >= 6:
						region = cur_idx - 1
					else:
						region = cur_idx

					match region:
						case 0:
							start_x = 0
						case 1:
							start_x = 1
						case 2:
							start_x = 2
						case 3 | 4:
							start_x = 3
						case 5 | 6:
							start_x = 6
						case 7:
							start_x = 7
						case 8:
							start_x = 8
						case _:
							start_x = 9

					if last_reg == region:
						if chr[last_y,last_x] == c:
							continue

						distance += path_map[chr[last_y,last_x]][c]
						last_x = cur_idx
						last_y = cur_idy
					else:
						last_reg = region
						last_x = cur_idx
						last_y = cur_idy
						if chr[1,start_x] == c:
							continue

						distance += path_map[chr[1,start_x]][c]


		scr_list.update({distance: p})

	return scr_list

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

# tests
pop = pop_init(10)

print(pop)

scr_list = calc_fitness(pop, 10)

print(scr_list)


