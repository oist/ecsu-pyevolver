"""
TODO: Missing module docstring
"""
from pyevolver.evolution import Evolution, MIN_SEARCH_VALUE, MAX_SEARCH_VALUE
from pyevolver import utils
import numpy as np
from pytictoc import TicToc
from numpy.random import RandomState
import os


def evaluate(population_genotype, _):
    range_input = (MIN_SEARCH_VALUE, MAX_SEARCH_VALUE)
    range_output = (-10, 10)
    performances = []
    for gen in population_genotype:
        gen = utils.linmap(gen, range_input, range_output)
        perf = 200 - (gen[0]**2 + gen[1]**2)
        performances.append(perf)
    return performances


def evolution_demo(folder_path):
    evo = Evolution(
        random_seed=123,
        population_size=1000,
        genotype_size=2,
        evaluation_function=evaluate,
        fitness_normalization_mode='RANK',
        selection_mode='RWS',
        reproduction_mode='HILL_CLIMBING',  #'GENETIC_ALGORITHM'
        mutation_variance=0.1,
        elitist_fraction=0.1,
        mating_fraction=0.9,
        crossover_probability=0.5,
        crossover_mode='1-POINT',
        max_generation=100,
        folder_path=folder_path,
        termination_function=None,
        checkpoint_interval=5
    )
    evo.run()




if __name__ == "__main__":
    folder_path = './tmp'
    utils.make_dir_if_not_exists(folder_path)
    evolution_demo(folder_path)
