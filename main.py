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

from PyQt5.QtCore import (
	QObject,
	QThread,
	pyqtSignal,
	pyqtSlot
)

from PyQt5.QtWidgets import QApplication, QMainWindow

from pyqtgraph import PlotWidget

class main_worker(QObject):

	# custom signals for updating gui
	update_gencount = pyqtSignal(int)
	update_currgen = pyqtSignal(float)
	update_keys = pyqtSignal(np.ndarray)
	update_lastgen = pyqtSignal(float)
	update_mingen = pyqtSignal(float)
	update_maxgen = pyqtSignal(float)
	update_plot = pyqtSignal(int, float)
	update_progress = pyqtSignal(int)

	send_saved = pyqtSignal(float, np.ndarray)

	finished = pyqtSignal()
	started = pyqtSignal()

	def __init__(self, gen_size):
		super(main_worker, self).__init__()
		self.rng = default_rng()
		self.__stop = False
		self.__end = False
		self.gen_size = gen_size

	@pyqtSlot()
	def work(self):
		self.started.emit()
		self.next_iter(self.gen_size)
		self.finished.emit()

	def stop(self):
		self.__stop = True
	

	# population initializer
	def pop_init(self, p_size):
		gene_pool = list("abcdefghijklmnopqrstuvwxyz[];\',./<\\") # [ ] ; ' \ , . / <

		# every layout has 35 keys total + padding
		pop = np.empty((0, 36), 'U')

		for _ in range(p_size):

			# gene_pool is 26 characters long and that leaves 9 for the symbols (extras)
			current = np.array(gene_pool, ndmin=2)

			self.rng.shuffle(current, axis=1)
			current = np.array(np.pad(current[0], (0, 1), constant_values="-"), ndmin=2)

			pop = np.concatenate((pop, current))

		return pop

	# as the name suggests it will calculate fitness score for each chromosome
	def calc_fitness(self, pop, p_size):
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
				self.update_progress.emit(int((100*(p+1))/p_size))
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
	def mutate(self, chr):
		# array length - 1 so it doesn't overflow
		rnd_int = self.rng.integers(0, 34)
		chr[[rnd_int, rnd_int + 1]] = chr[[rnd_int + 1, rnd_int]]

		return chr

	# main function to combine two chromosomes
	def crossover(self, chr1, chr2):

		rand_ind = self.rng.integers(0,34)
		rand_len = self.rng.integers(0,34)

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
	def next_iter(self, gen_size):

		p_size = 10
		pop = self.pop_init(p_size)

		lastgen_scr = 0
		maxgen_scr = 0
		self.mingen_scr = 9999999
		for i in range(gen_size):

			if self.__stop:
				self.send_saved.emit(mingen_scr, mingen_lyt)
				return

			self.update_gencount.emit(i+1)
			self.update_lastgen.emit(lastgen_scr)
			print("pass:", i)
			
			scr_list = self.calc_fitness(pop, p_size)
			scr_sorted = np.sort([x for x in scr_list.keys()])
			
			self.update_currgen.emit(scr_sorted[0])
			if scr_sorted[0] < self.mingen_scr:
				self.mingen_scr = scr_sorted[0]
				self.mingen_lyt = pop[scr_list[scr_sorted[0]]].copy()
				self.update_mingen.emit(self.mingen_scr)

			if scr_sorted[-1] > maxgen_scr:
				maxgen_scr = scr_sorted[-1]
				self.update_maxgen.emit(maxgen_scr)

			self.update_plot.emit(i, scr_sorted[0])
			self.update_keys.emit(pop[scr_list[scr_sorted[0]]][:-1])

			print("scr_sorted:",scr_sorted)
			new_pop = np.empty((0,36), 'U')

			for _ in range(p_size):
				chr1 = pop[scr_list[scr_sorted[0]]]
				chr2 = pop[scr_list[scr_sorted[1]]]

				p_cros = self.crossover(chr1, chr2)

				if (rand_int := self.rng.integers(0,10)) == 1:
					p_cros = self.mutate(p_cros)

				new_pop = np.concatenate((new_pop, np.array(p_cros, ndmin=2)))

			pop = new_pop
			lastgen_scr = scr_sorted[0]

if __name__ == "__main__":
	app = QApplication(sys.argv)

	MainWindow = window.setup_mainwindow()
	MainWindow.show()

	sys.exit(app.exec_())
