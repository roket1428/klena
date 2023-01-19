#	project-e find most efficient keyboard layout using genetic algorithm
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


def path_mapgen(p):
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

