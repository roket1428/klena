import numpy as np

from numpy.random import default_rng

def mapgen(p):
	# undirectional weighted graph for distance calculation
	path_map = {
				# left hand
				# pinky
				p[0,0]: { p[1,0]: 1.9636, p[2,0]: 3.8396 },
				p[1,0]: { p[0,0]: 1.9636, p[2,0]: 2.1298 },
				p[2,0]: { p[0,0]: 3.8396, p[1,0]: 2.1298 },

				# ring
				p[0,1]: { p[1,1]: 1.9636, p[2,1]: 3.8396 },
				p[1,1]: { p[0,1]: 1.9636, p[2,1]: 2.1298 },
				p[2,1]: { p[0,1]: 3.8396, p[1,1]: 2.1298 },

				# middle
				p[0,2]: { p[1,2]: 1.9636, p[2,2]: 3.8396 },
				p[1,2]: { p[0,2]: 1.9636, p[2,2]: 2.1298 },
				p[2,2]: { p[0,2]: 3.8396, p[1,2]: 2.1298 },

				# index
				p[0,3]: {
					p[1,3]: 1.9636, p[2,3]: 3.8396, p[0,4]: 1.9050,
					p[1,4]: 3.0495, p[2,4]: 4.0691, p[2,5]: 5.0626
				},
				p[1,3]: {
					p[0,3]: 1.9636, p[2,3]: 2.1298, p[0,4]: 2.3812,
					p[1,4]: 1.9050, p[2,4]: 2.1298, p[2,5]: 3.4343
				},
				p[2,3]: {
					p[0,3]: 3.8396, p[1,3]: 2.1298, p[0,4]: 4.4929,
					p[1,4]: 3.4343, p[2,4]: 1.9050, p[2,5]: 3.8100
				},
				p[0,4]: {
					p[0,3]: 1.9050, p[1,3]: 2.3812, p[2,3]: 4.4929,
					p[1,4]: 1.9636, p[2,4]: 3.8396, p[2,5]: 4.0691
				},
				p[1,4]: {
					p[0,3]: 3.0495, p[1,3]: 1.9050, p[2,3]: 3.4343,
					p[0,4]: 1.9636, p[2,4]: 2.1298, p[2,5]: 2.1298
				},
				p[2,4]: {
					p[0,3]: 4.0691, p[1,3]: 2.1298, p[2,3]: 1.9050,
					p[0,4]: 3.8396, p[1,4]: 2.1298, p[2,5]: 1.9050
				},

				# B key (hand agnostic)
				p[2,5]: {
					p[0,3]: 5.0626, p[1,3]: 3.4343, p[2,3]: 3.8100,
					p[0,4]: 4.0691, p[1,4]: 2.1298, p[2,4]: 1.9050,
					p[0,5]: 3.8396, p[1,5]: 2.1298, p[0,6]: 4.4929,
					p[1,6]: 3.4343, p[2,6]: 1.9050, p[2,7]: 3.8100
				},

				# right hand
				# index
				p[0,5]: {
					p[1,5]: 1.9636, p[0,6]: 1.9050, p[1,6]: 3.0495,
					p[2,6]: 4.0691, p[2,7]: 5.0626, p[2,5]: 3.8396
				},
				p[1,5]: {
					p[0,5]: 1.9636, p[0,6]: 2.3812, p[1,6]: 1.9050,
					p[2,6]: 2.1298, p[2,7]: 3.4343, p[2,5]: 2.1298
				},
				p[0,6]: {
					p[0,5]: 1.9050, p[1,5]: 2.3812, p[1,6]: 1.9636,
					p[2,6]: 3.8396, p[2,7]: 4.0691, p[2,5]: 4.4929
				},
				p[1,6]: {
					p[0,5]: 3.0495, p[1,5]: 1.9050, p[0,6]: 1.9636,
					p[2,6]: 2.1298, p[2,7]: 2.1298, p[2,5]: 3.4343
				},
				p[2,6]: {
					p[0,5]: 4.0691, p[1,5]: 2.1298, p[0,6]: 3.8396,
					p[1,6]: 2.1298, p[2,7]: 1.9050, p[2,5]: 1.9050
				},
				p[2,7]: {
					p[0,5]: 5.0626, p[1,5]: 3.4343, p[0,6]: 4.0691,
					p[1,6]: 2.1298, p[2,6]: 1.9050, p[2,5]: 3.8100
				},

				# middle
				p[0,7]: { p[1,7]: 1.9636, p[2,8]: 4.0691 },
				p[1,7]: { p[0,7]: 1.9636, p[2,8]: 2.1298 },
				p[2,8]: { p[0,7]: 4.0691, p[1,7]: 2.1298 },

				# ring
				p[0,8]: { p[1,8]: 1.9636, p[2,9]: 4.0691 },
				p[1,8]: { p[0,8]: 1.9636, p[2,9]: 2.1298 },
				p[2,9]: { p[0,8]: 4.0691, p[1,8]: 2.1298 },

				# pinky
				p[0,9]: {
					p[1,9]:  1.9636, p[0,10]: 1.9050, p[1,10]: 3.0495,
					p[2,10]: 4.0691, p[0,11]: 3.8100, p[1,11]: 4.6905
				},
				p[1,9]: {
					p[0,9]:  1.9636, p[0,10]: 2.3812, p[1,10]: 1.9050,
					p[2,10]: 2.1298, p[0,11]: 3.8396, p[1,11]: 3.8100
				},
				p[0,10]: {
					p[0,9]:  1.9050, p[1,9]:  2.3812, p[1,10]: 1.9636,
					p[2,10]: 3.8396, p[0,11]: 1.9050, p[1,11]: 3.0495
				},
				p[1,10]: {
					p[0,9]:  3.0495, p[1,9]:  1.9050, p[0,10]: 1.9636,
					p[2,10]: 2.1298, p[0,11]: 2.3812, p[1,11]: 1.9050
				},
				p[2,10]: {
					p[0,9]:  4.0691, p[1,9]:  2.1298, p[0,10]: 3.8396,
					p[1,10]: 2.1298, p[0,11]: 4.4929, p[1,11]: 3.4343
				},
				p[0,11]: {
					p[0,9]:  3.8100, p[1,9]:  3.8396, p[0,10]: 1.9050,
					p[1,10]: 2.3812, p[2,10]: 4.4929, p[1,11]: 1.9636
				},
				p[1,11]: {
					p[0,9]:  4.6905, p[1,9]:  3.8100, p[0,10]: 3.0495,
					p[1,10]: 1.9050, p[2,10]: 3.4343, p[0,11]: 1.9636
				}
	}

	return path_map

rng = default_rng()

gene_pool = list("abcdefghijklmnopqrstuvwxyz")
gene_ext = list(",.<>/?:;[]{}()|\\'\"")

pop = np.empty((0, 35), 'U')

text = "the quick brown fox jumps over the lazy dog"

for _ in range(10):
	current = np.array(np.concatenate((gene_pool, rng.choice(gene_ext, size=9, replace=False))), ndmin=2)
	rng.shuffle(current, axis=1)
	pop = np.concatenate((pop, current))

#     0   1   2   3   4   5   6   7   8   9   10  11
# 0 ['}' '.' 'j' 'a' 'h' 'o' 'y' 'b' 'w' 'n' 'm' 't']
# 1 ["'" 'v' 'f' 'g' 'p' 's' '/' '|' '[' 'r' 'e' 'i']
# 2 [';' 'd' 'l' 'q' 'k' ')' 'c' '?' 'z' 'u' 'x']

p = np.stack((pop[0,:12], pop[0,12:24], np.concatenate((pop[0,24:], [np.nan]))))

print(p)

path_map = mapgen(p)

distance = 0
last_reg = -1

for word in text.split():
	for c in word:

		print("char:", c)
		print("dist_before:", distance)
		cur_index_y = np.where(p==c)[0][0]
		cur_index_x = np.where(p==c)[1][0]

		if cur_index_y == 2 and cur_index_x >= 6:
			region = cur_index_x - 1
		else:
			region = cur_index_x

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
			print("same reg")
			if p[last_y,last_x] == c:
				print("same char, skipping")
				continue

			distance += path_map[p[last_y,last_x]][c]
			print("dist_after:", path_map[p[last_y,last_x]][c], distance)
			last_x = cur_index_x
			last_y = cur_index_y
		else:
			last_reg = region
			last_x = cur_index_x
			last_y = cur_index_y
			if p[1,start_x] == c:
				print("at start, skipping")
				continue

			distance += path_map[p[1,start_x]][c]
			print("dist_after:", path_map[p[1,start_x]][c], distance)

print(distance)
