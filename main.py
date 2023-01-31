#	project-e find the most efficient keyboard layout using the genetic algorithm
#		Copyright (C) 2023 roket1428 <meorhan@protonmail.com>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

# standart libraries
import mmap
import sys
import time

from multiprocessing import Process, Manager

# 3rd party libraries
import numpy as np
from numpy.random import default_rng

from smart_open import open

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QApplication

# local modules
import mapgen
import window

class MainWorker(QObject):

	# custom signals for updating gui
	updateGenCount = pyqtSignal(int)
	updateGenCurrent = pyqtSignal(float)
	updateGenLast = pyqtSignal(float)
	updateGenMin = pyqtSignal(float)
	updateGenMax = pyqtSignal(float)
	updateKeys = pyqtSignal(np.ndarray)
	updatePlot = pyqtSignal(int, float)
	updateProgressBar = pyqtSignal(int)

	sendSaved = pyqtSignal(float, np.ndarray)

	finished = pyqtSignal()
	started = pyqtSignal()

	def __init__(self, iter_size):
		super(MainWorker, self).__init__()
		self.iter_size = iter_size
		self.rng = default_rng()
		self.pop_size = 10
		self.__stop = False

	# start and stop functions for ui interaction
	@pyqtSlot()
	def work(self):
		self.started.emit()
		self.main_loop(self.iter_size)
		self.finished.emit()

	def stop(self):
		self.__stop = True

	def main_loop(self, iter_size):

		start = time.perf_counter()

		pop = self.pop_init()

		self.gen_score_min = sys.maxsize
		gen_score_max = 0
		gen_score_last = 0
		for i in range(iter_size):

			if self.__stop:
				self.sendSaved.emit(self.gen_score_min, self.gen_score_min_layout)
				return

			self.updateGenCount.emit(i+1)
			self.updateGenLast.emit(gen_score_last)
			print("pass:", i)

			score_dict = self.calc_fitness(pop)
			score_dict_keys_sorted = np.sort([x for x in score_dict.keys()])

			self.updateGenCurrent.emit(score_dict_keys_sorted[0])
			if score_dict_keys_sorted[0] < self.gen_score_min:
				self.gen_score_min = score_dict_keys_sorted[0]
				self.gen_score_min_layout = pop[score_dict[score_dict_keys_sorted[0]]].copy()
				self.updateGenMin.emit(self.gen_score_min)

			if score_dict_keys_sorted[-1] > gen_score_max:
				gen_score_max = score_dict_keys_sorted[-1]
				self.updateGenMax.emit(gen_score_max)

			self.updatePlot.emit(i, score_dict_keys_sorted[0])
			self.updateKeys.emit(pop[score_dict[score_dict_keys_sorted[0]]][:-1])

			print("score_dict_keys_sorted:",score_dict_keys_sorted)
			new_pop = np.empty((0,36), 'U')

			for _ in range(self.pop_size):
				chr1 = pop[score_dict[score_dict_keys_sorted[0]]]
				chr2 = pop[score_dict[score_dict_keys_sorted[1]]]

				p_cros = self.crossover(chr1, chr2)

				if self.rng.integers(0,10) == 1:
					p_cros = self.mutate(p_cros)

				new_pop = np.concatenate((new_pop, np.array(p_cros, ndmin=2)))

			pop = new_pop
			gen_score_last = score_dict_keys_sorted[0]

		finish = time.perf_counter()

		print(f"Finished in {round(finish-start)}s")

	# only called one time per run, creates a randomly generated population of self.pop_size
	def pop_init(self):

		gene_pool = list("abcdefghijklmnopqrstuvwxyz[];\',./<\\") # [ ] ; ' \ , . / <

		# every layout has 35 keys total + padding
		pop = np.empty((0, 36), 'U')

		for _ in range(self.pop_size):

			# gene_pool is 26 characters long and that leaves 9 for the symbols (extras)
			current = np.array(gene_pool, ndmin=2)

			self.rng.shuffle(current, axis=1)
			current = np.array(np.pad(current[0], (0, 1), constant_values="-"), ndmin=2)

			pop = np.concatenate((pop, current))

		return pop

	def calc_fitness(self, pop):

		biagram_map = mapgen.biagram_map

		# key pressing difficulity map according to experiments/assets/heatmap.png
		key_bias = np.stack([
							[  4, 2, 2, 2.5,  3.5,    5, 2.5,   2, 2, 3.5, 4.5, 4.5],
							[1.5, 1, 1,   1, 1.75, 1.75,   1,   1, 1, 1.5,   3,   3],
							[3.5, 4, 4, 2.5,  1.5,    2, 1.5, 2.5, 3,   4,   4,   0]
							])

		smf_bias = 10

		score_dict = {}

		processes = [None] * self.pop_size
		manager = Manager()
		scores = manager.list([None] * self.pop_size)

		with open("corpus/corpus_filtered.txt", "r") as f:
			corpus = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

			for p in range(self.pop_size):

				chr = np.array(pop[p].reshape(3,12), bytes)
				path_map = mapgen.path_mapgen(chr)

				cord_map = {}
				for i in range(chr.shape[0]):
					for j in range(chr.shape[1]):
						cord_map.update({chr[i,j]: { 0: i, 1: j}})

				processes[p] = Process(target=calc_score, args=(p, scores, chr, cord_map, corpus, path_map, biagram_map, key_bias, smf_bias))
				processes[p].start()

		for i, p in enumerate(processes):
			p.join()
			score_dict.update({scores[i]: i})
			self.updateProgressBar.emit(int((100*(i+1))/self.pop_size))

		return score_dict

	# when used, will swap two consecutive elements inside array, at a random index
	def mutate(self, chr):
		# array length - 1 so it doesn't overflow
		rnd_int = self.rng.integers(0, 34)
		chr[[rnd_int, rnd_int + 1]] = chr[[rnd_int + 1, rnd_int]]

		return chr

	def crossover(self, chr1, chr2):

		rnd_idx = self.rng.integers(0,34)
		rnd_len = self.rng.integers(0,34)

		offspring = chr1.copy()
		offspring[:35] = "_"

		for c in range(rnd_len):
			if rnd_idx > 34:
				rnd_idx = 0

			offspring[rnd_idx] = chr1[c]
			rnd_idx += 1

		chr2_idx = 0
		while "_" in offspring:
			if chr2_idx > 34:
				chr2_idx = 0
			if rnd_idx > 34:
				rnd_idx = 0

			if chr2[chr2_idx] in offspring:
				chr2_idx += 1
				continue

			offspring[rnd_idx] = chr2[chr2_idx]

			chr2_idx += 1
			rnd_idx += 1

		return offspring



def calc_score(index, scores, chr, cord_map, corpus, path_map, biagram_map, key_bias, smf_bias):

	print("pop:", index)
	score = 0
	last_region = last_y = last_x = last_hand = -1
	for char in corpus:

		cur_idy = cord_map[char][0]
		cur_idx = cord_map[char][1]

		if cur_idy == 2 and cur_idx >= 6:
			region = cur_idx - 1
		else:
			region = cur_idx

		if last_region == region:

			if chr[last_y,last_x] == char:
				score += (key_bias[last_y,last_x] + smf_bias)

			else:
				score += (path_map[chr[last_y,last_x]][char]*2 + key_bias[cur_idy,cur_idx] + smf_bias)
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
				biagram = f"{chr[last_y,last_x]}{char}"
				if biagram_map.get(biagram) != None and last_y == cur_idy:
					score -= (biagram_map[biagram]** 2)

			if chr[1,start_x] == char:
				score += key_bias[1,start_x]

			else:
				if last_hand != -1 and last_hand != cur_hand:
					score -= (path_map[chr[1,start_x]][char] * 0.5)

				score += (path_map[chr[1,start_x]][char]*2 + key_bias[cur_idy,cur_idx])

				last_hand = cur_hand
				last_region = region
				last_x = cur_idx
				last_y = cur_idy

	scores[index] = score


if __name__ == "__main__":
	app = QApplication(sys.argv)

	MainWindow = window.SetupMainwindow()
	MainWindow.show()

	sys.exit(app.exec_())
