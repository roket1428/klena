import unittest
import sys
import pathlib
import numpy as np

sys.path.insert(0, str(pathlib.PurePath(__file__).parent.parent))
from klena import main

class functions_TestCase(unittest.TestCase):
	def setUp(self):
		# headless main program
		self.prog = main.MainProgram("layout.txt", False)
		# test defined gene pool
		self.gene_pool = list("abcdefghijklmnopqrstuvwxyz[];',./<\\")

		# test call of the functions from the program
		self.pop = self.prog.pop_init()
		self.mutated_chr = self.pop[0].copy()
		self.prog.mutate(self.mutated_chr)
		self.p_cros = self.prog.crossover(self.pop[0], self.pop[1])
		self.score_dict = self.prog.calc_fitness(self.pop)

	def test_pop_init_chars(self):
		""" check for unwanted characters in pop_init output """
		for i in range(self.prog.pop_size):
			for j in range(self.prog.layout_size):
				self.assertIn(self.pop[i,j], self.gene_pool)

	def test_pop_init_shape(self):
		""" make sure pop_init's output is of the correct shape """
		self.assertEqual(self.pop.shape, (self.prog.pop_size, self.prog.layout_size+1))

	def test_pop_init_duplicate_chars(self):
		""" check for duplicate characters in pop_init output """
		for i in range(self.prog.pop_size):
			duplicates = []
			for j in range(self.prog.layout_size):
				if self.pop[i,j] not in duplicates:
					duplicates.append(self.pop[i,j])
				else:
					self.assertNotIn(self.pop[i,j], duplicates)

	def test_mutate_chars(self):
		""" check for unwanted characters in the mutated chr """
		for i in range(self.prog.layout_size):
			self.assertIn(self.mutated_chr[i], self.gene_pool)

	def test_mutate_duplicate_chars(self):
		""" check for duplicate characters in the mutated chr """
		duplicates = []
		for i in range(self.prog.layout_size):
			if self.mutated_chr[i] not in duplicates:
				duplicates.append(self.mutated_chr[i])
			else:
				self.assertNotIn(self.mutated_chr[i], duplicates)

	def test_mutate_output_difference(self):
		""" mutated chr should't be equal to original chr (pop[0] in this case) """
		self.assertFalse((self.mutated_chr==self.pop[0]).all())

	def test_crossover_chars(self):
		""" check for unwanted characters in the crossover output """
		for i in range(self.prog.layout_size):
			self.assertIn(self.p_cros[i], self.gene_pool)

	def test_crossover_duplicates(self):
		""" check for duplicate characters in the crossover output """
		duplicates = []
		for i in range(self.prog.layout_size):
			if self.p_cros[i] not in duplicates:
				duplicates.append(self.p_cros[i])
			else:
				self.assertNotIn(self.p_cros[i], duplicates)

	def test_crossover_difference(self):
		""" crossover output should't be equal to either of the parents (pop[0] or pop[1] in this case) """
		self.assertFalse((self.p_cros==self.pop[0]).all())
		self.assertFalse((self.p_cros==self.pop[1]).all())

	def test_calc_fitness_types(self):
		""" calc_fitness' output types should match with the pre-defined ones """
		for i in range(self.prog.pop_size):
			self.assertEqual(type(list(self.score_dict.keys())[i]), np.float64)
			self.assertEqual(type(list(self.score_dict.values())[i]), int)

if __name__ == "__main__":
	unittest.main()
