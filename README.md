# project-e [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Find the most efficient keyboard layout using the genetic algorithm.

## How It Works?
The project uses genetic algorithms' main principles.
- Create a population consisting of 10 randomly generated layouts from the gene pool (abcdefghijklmnopqrstuvwxyz[];',./<\)
- Calculate a fitness score for each layout.
- Create 10 offspring from the two fittest layouts (the lower the fitness score, the better).
- Calculate the fitness score for the new generation and continue to the last step.

### Calculating The Fitness Score
Fitness score calculation simulates key presses while looping through the dataset.
- First, it calculates the finger travel distance by using a hard-coded weighted graph (assuming 8 fingers at the home row).
- Then it checks for the [biagrams](https://en.wikipedia.org/wiki/Bigram "biagrams") and reduces the score accordingly.
- After that the key bias (according to the physical locations of the keys) and the same finger bias (repetitive key presses) is added.

## Status
- Most features are implemented.
- GUI and the main logic is working.

## TODOs
- [ ] Normalize and balance the fitness score.
- [x] <s>Reduce the dataset size by using proper dimension reduction techniques.</s>
- [x] <s>Multi-core implementation of the fitness calculation function.</s>
- [ ] Fix indentation and naming inconsistencies.
- [ ] Handle the command line arguments.
