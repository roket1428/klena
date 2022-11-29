#import cProfile, pstats, io
#from pstats import SortKey

import mapgen

import numpy as np

from numpy.random import default_rng
from smart_open import open
from collections import Counter

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
	gene_ext = list(",.<>/?:;[]{}\\()|\'\"")

	# every layout has 35 keys total
	pop = np.empty((0, 35), 'U')

	for _ in range(p_size):

		# gene_pool is 26 characters long and that leaves 9 for the symbols (extras)
		current = np.array(np.concatenate((gene_pool, rng.choice(gene_ext, size=9, replace=False))), ndmin=2)

		rng.shuffle(current, axis=1)
		pop = np.concatenate((pop, current))

	return pop


# as the name suggests it will calculate fitness score for each chromosome
#@profile
def calc_fitness(pop, p_size):

	# biagram list
	biag_map = mapgen.biag_map

	# key pressing difficulity map according to experiments/assets/heatmap.png
	key_bias = np.stack([
						[  4, 2, 2, 2.5,  3.5,    5, 2.5,   2, 2, 3.5, 4.5, 4.5],
						[1.5, 1, 1,   1, 1.75, 1.75,   1,   1, 1, 1.5,   3,   3],
						[3.5, 4, 4, 2.5,  1.5,    2, 1.5, 2.5, 3,   4,   4,   0]
						])

	# same finger biagram bias
	smf_bias = 10

	scr_list = {}

	# for every population member
	for p in range(p_size):

		print("pop:", p)
		corpus = Corpus()
		chr = np.append(pop[p], [np.nan]).reshape(3,12)

		path_map = mapgen.path_mapgen(chr)

		chr_cord = {}
		for i in range(chr.shape[0]):
			for j in range(chr.shape[1]):
				chr_cord.update({chr[i,j]: { 0: i, 1: j}})

		distance = 0
		last_reg = last_y = last_x = last_hand = -1
		for doc in corpus:
			print("total dist:", distance)

			for w in doc:
				for c in w:

					cur_idy = chr_cord[c][0]
					cur_idx = chr_cord[c][1]

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
							if biag_map.get(biag) != None and last_y == cur_idy:
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

	# 0 left, 1 right
	# side = rng.integers(0,2)
	# print("side:", side)
	side = 0

	gene_pool = chr1.ravel()

	chr1_lhalf = np.stack((chr1[0,:5], chr1[1,:5], chr1[2,:5]))
	chr1_rhalf = np.stack((chr1[0,5:], chr1[1,5:], chr1[2,5:]))

	chr2_lhalf = np.stack((chr2[0,:5], chr2[1,:5], chr2[2,:5]))
	chr2_rhalf = np.stack((chr2[0,5:], chr2[1,5:], chr2[2,5:]))

	if side == 1:
		rem_pool = chr1_rhalf.ravel()
		chr2_pool = np.empty((0,2), dtype='U')
		chr1_rhalf_out = chr1_rhalf.copy()
		chr1_rhalf_out[:] = "_"

		for i in range(chr1_rhalf.shape[0]):
			for j in range(chr1_rhalf.shape[1]):
				if chr2_rhalf[i,j] in rem_pool:
					chr1_rhalf_out[i,j] = chr2_rhalf[i,j]
					chr2_pool = np.append(chr2_pool, chr2_rhalf[i,j])

		diff_pool = [x for x in rem_pool if x not in chr2_pool]
		for i in range(chr1_rhalf.shape[0]):
			for j in range(chr1_rhalf.shape[1]):
				if chr1_rhalf_out[i,j] == "_":
					chr1_rhalf_out[i,j] = diff_pool[rng.integers(0, len(diff_pool))]
					diff_pool.pop(diff_pool.index(chr1_rhalf_out[i,j]))
		chr1_rhalf = chr1_rhalf_out
	else:

		rem_pool = chr1_lhalf.ravel()
		chr2_pool = np.empty((0,2), dtype='U')
		chr1_lhalf_out = chr1_lhalf.copy()
		chr1_lhalf_out[:] = "_"

		for i in range(chr1_lhalf.shape[0]):
			for j in range(chr1_lhalf.shape[1]):
				if chr2_lhalf[i,j] in rem_pool:
					chr1_lhalf_out[i,j] = chr2_lhalf[i,j]
					chr2_pool = np.append(chr2_pool, chr2_lhalf[i,j])

		diff_pool = [x for x in rem_pool if x not in chr2_pool]
		for i in range(chr1_lhalf.shape[0]):
			for j in range(chr1_lhalf.shape[1]):
				if chr1_lhalf_out[i,j] == "_":
					chr1_lhalf_out[i,j] = diff_pool[rng.integers(0, len(diff_pool))]
					diff_pool.pop(diff_pool.index(chr1_lhalf_out[i,j]))
		chr1_lhalf = chr1_lhalf_out


	return np.concatenate((chr1_lhalf, chr1_rhalf), axis=1)

# create next generation by combining fittest chromosomes
def next_iter():
	pass

# tests
pop = pop_init(10)
print(pop)

chr1 = np.append(pop[0], [np.nan]).reshape(3,12)
chr2 = np.append(pop[1], [np.nan]).reshape(3,12)

print("crossover testing")
co = crossover(chr1, chr2)
print(chr1)
print(f"\n{chr2}\n")
print(co)
print(Counter(co.ravel().tolist()))

# crossover test
# for i in range(10000):
# 	print("i:",i)
# 	pop = pop_init(10)
# 	for _ in range(10):
# 		chr1 = np.append(pop[rng.integers(0,10)], [np.nan]).reshape(3,12)
# 		chr2 = np.append(pop[rng.integers(0,10)], [np.nan]).reshape(3,12)
# 		co = crossover(chr1, chr2)
# 		for x in Counter(co.ravel().tolist()).values():
# 			if x != 1:
# 				print(Counter(co.ravel().tolist()))
# 				raise AssertionError


#scr_list = calc_fitness(pop, 10)
#
#print(scr_list)

#pr.disable()
#s = io.StringIO()
#sortby = SortKey.CUMULATIVE
#ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
#ps.print_stats()
#print(s.getvalue())

