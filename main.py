# local modules
import gui
import mapgen
import window

# standart modules
import mmap
import sys

# 3rd party libs
import numpy as np

from numpy.random import default_rng
from smart_open import open

from PyQt5 import QtCore, QtGui, QtWidgets
from pyqtgraph import PlotWidget


rng = default_rng()

# population initializer
def pop_init(p_size):
	gene_pool = list("abcdefghijklmnopqrstuvwxyz[];\',./<\\") # [ ] ; ' \ , . / <

	# every layout has 35 keys total + padding
	pop = np.empty((0, 36), 'U')

	for _ in range(p_size):

		# gene_pool is 26 characters long and that leaves 9 for the symbols (extras)
		current = np.array(gene_pool, ndmin=2)

		rng.shuffle(current, axis=1)
		current = np.array(np.pad(current[0], (0, 1), constant_values="-"), ndmin=2)

		pop = np.concatenate((pop, current))

	return pop


# as the name suggests it will calculate fitness score for each chromosome
#@profile
def calc_fitness(ui, pop, p_size):

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

	with open("corpus/corpus_minimal_stripped.txt", "r") as f:
		corpus = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

		# for every population member
		for p in range(p_size):

			print("pop:", p)
			ui.update_progress.emit(int((100*(p+1))/p_size))
			#corpus = Corpus()
			chr = np.array(pop[p].reshape(3,12), bytes)

			path_map = mapgen.path_mapgen(chr)

			chr_cord = {}
			for i in range(chr.shape[0]):
				for j in range(chr.shape[1]):
					chr_cord.update({chr[i,j]: { 0: i, 1: j}})

			distance = 0
			last_reg = last_y = last_x = last_hand = -1
			for c in corpus:
				cur_idy = chr_cord[c][0]
				cur_idx = chr_cord[c][1]

				if cur_idy == 2 and cur_idx >= 6:
					region = cur_idx - 1
				else:
					region = cur_idx

				if last_reg == region:

					if chr[last_y,last_x] == c:
						distance += (key_bias[last_y,last_x] + smf_bias)
						continue

					distance += (path_map[chr[last_y,last_x]][c]*2 + key_bias[cur_idy,cur_idx] + smf_bias)
					last_x = cur_idx
					last_y = cur_idy
				else:

					if region < 5:
						cur_hand = 0
					else:
						cur_hand = 1

					if region == 4:
						start_x = 3
					elif region == 5:
						start_x = 6
					elif region > 9:
						start_x = 9
					else:
						start_x = region

					if last_y != -1:
						biag = f"{chr[last_y,last_x]}{c}"
						if biag_map.get(biag) != None and last_y == cur_idy:
							distance -= (biag_map[biag]** 2)

					if chr[1,start_x] == c:
						distance += key_bias[1,start_x]
						continue

					if last_hand != -1 and last_hand != cur_hand:
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

	rand_ind = rng.integers(0,34)
	rand_len = rng.integers(0,34)

	chr_out = chr1.copy()
	chr_out[:35] = "_"

	for c in range(rand_len):
		if rand_ind > 34:
			rand_ind = 0

		chr_out[rand_ind] = chr1[c]
		rand_ind += 1

	chr2_index = 0
	while "_" in chr_out:
		if chr2_index > 34:
			chr2_index = 0
		if rand_ind > 34:
			rand_ind = 0

		if chr2[chr2_index] in chr_out:
			chr2_index += 1
			continue

		chr_out[rand_ind] = chr2[chr2_index]

		chr2_index += 1
		rand_ind += 1

	return chr_out

# create next generation by combining fittest chromosomes
def next_iter(ui, gen_size):

	p_size = 10
	pop = pop_init(p_size)

	lastgen_scr = 0
	maxgen_scr = 0
	mingen_scr = 9999999
	for i in range(gen_size):

		if ui.quit_flag:
			print("finished")
			return

		ui.update_gencount.emit(i+1)
		ui.update_lastgen.emit(lastgen_scr)
		print("pass:", i)
		
		scr_list = calc_fitness(ui, pop, p_size)
		scr_sorted = np.sort([x for x in scr_list.keys()])
		
		ui.update_currgen.emit(scr_sorted[0])
		if scr_sorted[0] < mingen_scr:
			mingen_scr = scr_sorted[0]
			ui.update_mingen.emit(mingen_scr)

		if scr_sorted[-1] > maxgen_scr:
			maxgen_scr = scr_sorted[-1]
			ui.update_maxgen.emit(maxgen_scr)

		ui.update_plot.emit(i, scr_sorted[0])
		ui.update_keys.emit(pop[scr_list[scr_sorted[0]]][:-1])

		print("scr_sorted:",scr_sorted)
		new_pop = np.empty((0,36), 'U')

		for _ in range(p_size):
			chr1 = pop[scr_list[scr_sorted[0]]]
			chr2 = pop[scr_list[scr_sorted[1]]]

			p_cros = crossover(chr1, chr2)

			if (rand_int := rng.integers(0,10)) == 1:
				p_cros = mutate(p_cros)

			new_pop = np.concatenate((new_pop, np.array(p_cros, ndmin=2)))

		pop = new_pop
		lastgen_scr = scr_sorted[0]

if __name__ == "__main__":
	app = QtWidgets.QApplication(sys.argv)
	MainWindow = QtWidgets.QMainWindow()
	ui = window.setup_mainwindow(MainWindow, next_iter)
	MainWindow.show()
	sys.exit(app.exec_())

#pop = pop_init(10)
#co = crossover(pop[0], pop[1])
#
#print("chr1:\n", pop[0])
#print("chr2:\n", pop[1])
#print("chr_out:\n", co)

#print("crossover testing")
#co = crossover(chr1, chr2)
#print(chr1)
#print(f"\n{chr2}\n")
#print(co)
#print(Counter(co.ravel().tolist()))

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

