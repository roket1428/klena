import numpy as np
import random as rnd
# going to add numba or cython if the project becomes too complex

# population initializer func
def pop_init(p_size):
	pass

# as the name suggests it will calculate fitness score for each chromosome
def calc_fitness(p):
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
eng_gene_pool = np.array(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"])
alphabet_ext = np.array([",", ".", "<", ">", "/", "?", ":", ";", "[", "]", "{", "}", "\'", "\"", "\\", "|"])


