#	klena: find the most efficient keyboard layout using the genetic algorithm
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
import argparse
import mmap
import logging
import pathlib
import sys
import time

from multiprocessing import Process, Manager

# 3rd party libraries
import numpy as np
from numpy.random import default_rng

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QApplication

sys.path.insert(0, str(pathlib.PurePath(__file__).parent.parent))
from klena import mapgen, window

class MainProgram(object):
	""" main class of the program, runs standalone in headless mode and used as a template in gui mode """
	def __init__(self, output=None, auto_save=False):
		super(MainProgram, self).__init__()
		self.pop_size = 10
		self.layout_size = 35
		self.output = output
		self.auto_save = auto_save
		self.log = logging.getLogger("__main__")
		self.rng = default_rng()

	def action_save_prompt(self, i, val):
		if not self.auto_save:
			ans = input(f"save layout {val}? [y/N]")
			if ans.lower() != "y":
				return
		self.log.info(f"saving layout {self.gen_score_min} to {pathlib.Path(self.output).absolute()}")
		with open(self.output, "a", encoding='utf-8') as f:
			f.write(f"generations: {i}\tscore: {round(self.gen_score_min)}\n")
			f.write(f"best layout:\n{self.gen_score_min_layout.reshape(3,12)}\n")

	# reserved template actions for the gui (will be overridden by MainWorker)
	def action_gen_count(self, val):
		return

	def action_gen_last(self, val):
		return

	def action_gen_current(self, val):
		return

	def action_gen_min(self, val):
		return

	def action_gen_max(self, val):
		return

	def action_saved(self):
		return False

	def action_plot(self, i, val):
		return

	def action_keys(self, val):
		return

	def action_progressbar(self, val):
		return

	def run(self, iter_size, dataset):
		self.log.info("started")
		start_timer = time.perf_counter()

		pop = self.pop_init()
		gen_score_max = gen_score_last = 0
		self.gen_score_min = sys.maxsize
		score_dict_keys_sorted = np.array([0]* self.pop_size)

		for i in range(iter_size):
			if self.action_saved():
				return

			self.action_gen_count(i+1)
			self.log.info(f"iter: {i+1}")

			if i > 0:
				gen_score_last = score_dict_keys_sorted[0]

			score_dict = self.calc_fitness(pop, dataset)
			# 1d numpy array with sorted scores (ascending)
			score_dict_keys_sorted = np.sort(list(score_dict.keys()))

			self.log.debug(f"last min:\t{gen_score_last}")
			self.action_gen_last(gen_score_last)

			self.log.debug(f"current min:\t{score_dict_keys_sorted[0]}")
			self.action_gen_current(score_dict_keys_sorted[0])

			if score_dict_keys_sorted[0] < self.gen_score_min:
				self.gen_score_min = score_dict_keys_sorted[0]
				self.gen_score_min_layout = pop[score_dict[score_dict_keys_sorted[0]]].copy()

			self.log.debug(f"all time min:\t{self.gen_score_min}")
			self.action_gen_min(self.gen_score_min)

			if score_dict_keys_sorted[-1] > gen_score_max:
				gen_score_max = score_dict_keys_sorted[-1]
			self.log.debug(f"all time max:\t{gen_score_max}")
			self.action_gen_max(gen_score_max)

			self.action_plot(i, score_dict_keys_sorted[0])
			self.action_keys(pop[score_dict[score_dict_keys_sorted[0]]][:-1])
			self.log.debug(f"score sorted: {score_dict_keys_sorted}")

			new_pop = np.empty((0, self.layout_size+1), 'U')
			for _ in range(self.pop_size):
				chr1 = pop[score_dict[score_dict_keys_sorted[0]]]
				chr2 = pop[score_dict[score_dict_keys_sorted[1]]]
				p_cros = self.crossover(chr1, chr2)
				if self.rng.integers(0,10) == 1:
					self.mutate(p_cros)

				new_pop = np.concatenate((new_pop, np.array(p_cros, ndmin=2)))

			pop = new_pop

		end_timer = time.perf_counter()
		self.log.info("stopped")
		self.log.debug(f"finished in {end_timer-start_timer}s")
		self.action_save_prompt(iter_size, self.gen_score_min)

	def pop_init(self):
		""" only called one time per run, creates a randomly generated population of pop_size """
		# alphabet is 26 characters long that leaves 9 for the symbols (extras)
		gene_pool = list("abcdefghijklmnopqrstuvwxyz[];\',./<\\") # [ ] ; ' \ , . / <

		# array structure: [["d", "x", "n"...], ["e","/", "b"...]...] 2d numpy array of shape (0, 36), 35 keys + 1 padding
		pop = np.empty((0, self.layout_size+1), 'U')

		for _ in range(self.pop_size):
			current = np.array(gene_pool, ndmin=2)

			self.rng.shuffle(current, axis=1)
			current = np.array(np.pad(current[0], (0, 1), constant_values="-"), ndmin=2)

			pop = np.concatenate((pop, current))

		return pop

	def calc_fitness(self, pop, dataset):
		""" calculates fitness score for each member of the current population """

		# predefined biagram map from mapgen
		biagram_map = mapgen.biagram_map
		# key pressing difficulity map according to assets/heatmap.png
		key_bias = np.stack([
							[  4, 2, 2, 2.5,  3.5,    5, 2.5,   2, 2, 3.5, 4.5, 4.5],
							[1.5, 1, 1,   1, 1.75, 1.75,   1,   1, 1, 1.5,   3,   3],
							[3.5, 4, 4, 2.5,  1.5,    2, 1.5, 2.5, 3,   4,   4,   0]
							])
		# same finger bias, TODO: find a value which represents its weight better
		smf_bias = 10

		# dict structure {'score of the layout': 'index of the layout in pop[]'} -> {float: int}
		score_dict = {}

		# we initalize a manager and create a shared resource to be used among processes
		manager = Manager()
		scores = manager.list([None] * self.pop_size)
		processes = [None] * self.pop_size
		with open(dataset, "r") as f:
			corpus = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

			for p in range(self.pop_size):

				chrm = np.array(pop[p].reshape(3,12), bytes)
				path_map = mapgen.path_mapgen(chrm)

				cord_map = {}
				for i in range(chrm.shape[0]):
					for j in range(chrm.shape[1]):
						cord_map.update({chrm[i,j]: { 0: i, 1: j}})

				self.log.debug(f"pop: {p}")
				processes[p] = Process(target=calc_score, args=(p, scores, chrm, corpus, cord_map, path_map, biagram_map, key_bias, smf_bias))
				processes[p].start()

		for i, p in enumerate(processes):
			p.join()
			score_dict.update({scores[i]: i})
			self.action_progressbar(int((100*(i+1))/self.pop_size))

		return score_dict

	def mutate(self, chr1):
		""" swap two consecutive elements inside an array, at a random index """
		rnd_int = self.rng.integers(0, self.layout_size-1)
		chr1[[rnd_int, rnd_int + 1]] = chr1[[rnd_int + 1, rnd_int]]

	def crossover(self, chr1, chr2):
		""" combines most efficient two layouts (index 0 and 1) into one and returns it """
		rnd_idx = self.rng.integers(0, self.layout_size-1)
		rnd_len = self.rng.integers(0, self.layout_size-1)

		offspring = chr1.copy()
		offspring[:self.layout_size] = "_"

		for c in range(rnd_len):
			if rnd_idx > self.layout_size-1:
				rnd_idx = 0

			offspring[rnd_idx] = chr1[c]
			rnd_idx += 1

		chr2_idx = 0
		while "_" in offspring:
			if chr2_idx > self.layout_size-1:
				chr2_idx = 0
			if rnd_idx > self.layout_size-1:
				rnd_idx = 0

			if chr2[chr2_idx] in offspring:
				chr2_idx += 1
				continue

			offspring[rnd_idx] = chr2[chr2_idx]

			chr2_idx += 1
			rnd_idx += 1

		return offspring

class MainWorker(QObject, MainProgram):
	""" main class for the GUI thread """

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

	def __init__(self, iter_size, dataset):
		super(MainWorker, self).__init__()
		self.dataset = dataset
		self.iter_size = iter_size
		self.__stop = False

	# start and stop functions for ui interaction
	@pyqtSlot()
	def work(self):
		self.started.emit()
		self.run(self.iter_size, self.dataset)
		self.finished.emit()

	def stop(self):
		self.__stop = True

	def action_saved(self):
		if self.__stop:
			self.sendSaved.emit(self.gen_score_min, self.gen_score_min_layout)
			return True

		return False

	def action_gen_count(self, val):
		self.updateGenCount.emit(val)

	def action_gen_last(self, val):
		self.updateGenLast.emit(val)

	def action_gen_current(self, val):
		self.updateGenCurrent.emit(val)

	def action_gen_min(self, val):
		self.updateGenMin.emit(val)

	def action_gen_max(self, val):
		self.updateGenMax.emit(val)

	def action_plot(self, i, val):
		self.updatePlot.emit(i, val)

	def action_keys(self, val):
		self.updateKeys.emit(val)

	def action_progressbar(self, val):
		self.updateProgressBar.emit(val)

	# save prompt will not be used in the gui
	def action_save_prompt(self, i, val):
		return

def calc_score(index, scores, chrm, corpus, cord_map, path_map, biagram_map, key_bias, smf_bias):
	""" calculates the fitness score for given pop member, it will run on a seperate process and after it finishes will update the shared resource 'scores' """
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
			if chrm[last_y,last_x] == char:
				score += (key_bias[last_y,last_x] + smf_bias)
			else:
				score += (path_map[chrm[last_y,last_x]][char]*2 + key_bias[cur_idy,cur_idx] + smf_bias)
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
				biagram = f"{chrm[last_y,last_x]}{char}"
				if biagram_map.get(biagram) != None and last_y == cur_idy:
					score -= (biagram_map[biagram]** 2)

			if chrm[1,start_x] == char:
				score += key_bias[1,start_x]

			else:
				if last_hand != -1 and last_hand != cur_hand:
					score -= (path_map[chrm[1,start_x]][char] * 0.5)

				score += (path_map[chrm[1,start_x]][char]*2 + key_bias[cur_idy,cur_idx])

				last_hand = cur_hand
				last_region = region
				last_x = cur_idx
				last_y = cur_idy

	scores[index] = score

if __name__ == "__main__":

	parser = argparse.ArgumentParser(
			prog="klena",
			description="Find the most efficient keyboard layout using the genetic algorithm.")

	parser.add_argument('-hl', '--headless', action='store_true', help='run the program without gui')
	parser.add_argument('-v', '--verbose', action='store_true', help='print additional debugging information')
	parser.add_argument('iter_size', type=int, nargs='?', default=0, help='number of iterations')
	parser.add_argument('-y', '--auto-save', action='store_true', help='auto save the layout without prompt')
	parser.add_argument('-o', '--output', type=str, default="layout.txt", help='save path (default=layout.txt)')
	parser.add_argument('-i', '--dataset', type=str, default="corpus/corpus.txt", help='corpus path (default=corpus/corpus.txt)')
	args = parser.parse_args()

	output_path = pathlib.PurePath(args.output).parent
	pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

	log = logging.getLogger(__name__)
	log.setLevel(logging.DEBUG)

	pathlib.Path("logs").mkdir(exist_ok=True)
	log_file_handler = logging.FileHandler(f"logs/log-{round(time.time())}.txt", mode='w', encoding='utf-8')
	log_stream_handler = logging.StreamHandler(stream=sys.stdout)

	formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
	log_file_handler.setFormatter(formatter)
	log_stream_handler.setFormatter(formatter)

	log_file_handler.setLevel(logging.INFO)
	log_stream_handler.setLevel(logging.INFO)

	if args.verbose:
		log_file_handler.setLevel(logging.DEBUG)
		log_stream_handler.setLevel(logging.DEBUG)

	log.addHandler(log_file_handler)
	log.addHandler(log_stream_handler)

	if args.headless:
		if not args.iter_size:
			parser.error("number of iterations must be specified in headless mode")
			sys.exit(1)

		prog = MainProgram(output=args.output, auto_save=args.auto_save)
		prog.run(args.iter_size, args.dataset)

		log_stream_handler.close()
		log_file_handler.close()

	else:
		app = QApplication(sys.argv)

		MainWindow = window.SetupMainwindow(iter_size=args.iter_size, dataset=args.dataset, output=args.output, auto_save=args.auto_save)
		MainWindow.show()

		sys.exit(app.exec_())

