#import cProfile, pstats, io
#from pstats import SortKey

import pathmap

import numpy as np

from numpy.random import default_rng
from smart_open import open

#pr = cProfile.Profile()
#pr.enable()

class Corpus:
	def __iter__(self):
		for line in open("corpus/corpus_minimal.txt"):
			yield line.split()

rng = default_rng()

# population initializer
def pop_init(p_size):
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
def calc_fitness(pop, p_size):

	# biagram list
	biag_map = {
		"th": 3.56, "of": 1.17, "io": 0.83,
		"he": 3.07, "ed": 1.17, "le": 0.83,
		"in": 2.43, "is": 1.13, "ve": 0.83,
		"er": 2.05, "it": 1.12, "co": 0.79,
		"an": 1.99, "al": 1.09, "me": 0.79,
		"re": 1.85, "ar": 1.07, "de": 0.76,
		"on": 1.76, "st": 1.05, "hi": 0.76,
		"at": 1.49, "to": 1.05, "ri": 0.73,
		"en": 1.45, "nt": 1.04, "ro": 0.73,
		"nd": 1.35, "ng": 0.95, "ic": 0.70,
		"ti": 1.34, "se": 0.93, "ne": 0.69,
		"es": 1.34, "ha": 0.93, "ea": 0.69,
		"or": 1.28, "as": 0.87, "ra": 0.69,
		"te": 1.20, "ou": 0.87, "ce": 0.65
	}

	biag_keys = np.array([
		"th", "of", "io",
		"he", "ed", "le",
		"in", "is", "ve",
		"er", "it", "co",
		"an", "al", "me",
		"re", "ar", "de",
		"on", "st", "hi",
		"at", "to", "ri",
		"en", "nt", "ro",
		"nd", "ng", "ic",
		"ti", "se", "ne",
		"es", "ha", "ea",
		"or", "as", "ra",
		"te", "ou", "ce"
	])

	# key pressing difficulity map according to experiments/assets/heatmap.png
	key_bias = np.stack([
						[  4, 2, 2, 2.5,  3.5,    5, 2.5,   2, 2, 3.5, 4.5, 4.5],
						[1.5, 1, 1,   1, 1.75, 1.75,   1,   1, 1, 1.5,   3,   3],
						[3.5, 4, 4, 2.5,  1.5,    2, 1.5, 2.5, 3,   4,   4,   0]
						])

	# same finger biagram bias
	smf_bias = 10

	scr_list = {}

	for p in range(p_size):

		print("pop:", p)
		corpus = Corpus()
		chr = np.stack((pop[p,:12], pop[p,12:24], np.concatenate((pop[p,24:], np.array([np.nan])))))
		path_map = pathmap.mapgen(chr)

		distance = 0
		last_reg = last_y = last_x = last_hand = -1
		for ind, doc in enumerate(corpus):
			print("corpus: ", ind)
			print("total dist:", distance)
			for w in doc:
				for c in w:

					# ignore the genes that are not in the pool for now
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
							cur_hand = 0
						case 1:
							start_x = 1
							cur_hand = 0
						case 2:
							start_x = 2
							cur_hand = 0
						case 3 | 4:
							start_x = 3
							cur_hand = 0
						case 5 | 6:
							start_x = 6
							cur_hand = 1
						case 7:
							start_x = 7
							cur_hand = 1
						case 8:
							start_x = 8
							cur_hand = 1
						case _:
							start_x = 9
							cur_hand = 1

					if last_reg == region:

						if chr[last_y,last_x] == c:
							distance += (key_bias[last_y,last_x] + smf_bias)
							continue

						distance += (path_map[chr[last_y,last_x]][c]*2 + key_bias[cur_idy,cur_idx] + smf_bias)
						last_x = cur_idx
						last_y = cur_idy
					else:
						if last_y != -1:
							biag = f"{chr[last_y,last_x]}{c}"
							if biag in biag_keys and last_y == cur_idy:
								distance -= (biag_map[biag]** 2)

						if chr[1,start_x] == c:
							distance += key_bias[1,start_x]
							continue

						if last_hand != cur_hand and last_hand != -1:
							distance -= (path_map[chr[1,start_x]][c] * 0.5)

						distance += (path_map[chr[1,start_x]][c]*2 + key_bias[cur_idy,cur_idx])

						last_hand = cur_hand
						last_reg = region
						last_x = cur_idx
						last_y = cur_idy

		scr_list.update({distance: p})

	return scr_list

# when used, will swap some genes
def mutate(chr):
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

#pr.disable()
#s = io.StringIO()
#sortby = SortKey.CUMULATIVE
#ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
#ps.print_stats()
#print(s.getvalue())

