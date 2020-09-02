"""
TODO: Missing module docstring
"""
from evolution import Evolution, MIN_SEARCH_VALUE, MAX_SEARCH_VALUE
import utils
import numpy as np
from pytictoc import TicToc
from numpy.random import RandomState
import os


def rws_test():

    size = 10000
    selection = 1000
    random_state = RandomState()
    probs = random_state.uniform(size=size)
    probs /= sum(probs)

    random_state.seed(5)

    def standard_method():
        t.tic()
        result = []
        cum_probs = np.cumsum(probs)
        for _ in range(selection):
            r = random_state.random()
            for i in range(size):
                if r <= cum_probs[i]:
                    result.append(i)
                    break
        return result

    def numpy_method():
        return random_state.choice(size, size=selection, replace=True, p=probs)

    t = TicToc()
    t.tic()
    result_standard_method = standard_method()
    elp_std = t.tocvalue(restart=True)
    result_numpy_method = numpy_method()
    elp_np = t.tocvalue()
    print('standard: {}'.format(elp_std))
    print('numpy: {}'.format(elp_np))
    print(result_numpy_method)
    print(result_standard_method)


def evaluate(population_genotype, _):
    range_input = (MIN_SEARCH_VALUE, MAX_SEARCH_VALUE)
    range_output = (-10, 10)
    performances = []
    for gen in population_genotype:
        gen = utils.linmap(gen, range_input, range_output)
        perf = 200 - (gen[0]**2 + gen[1]**2)
        performances.append(perf)
    return performances


def test_evolution():
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
        folder_path='./tmp',
        termination_function=None,
        checkpoint_interval=5
    )
    evo.run()


def time_evolution():
    evo = Evolution(
        random_seed=123,
        population_size=5000,
        genotype_size=100,
        evaluation_function=lambda l: [1]*len(l),
        fitness_normalization_mode='RANK',
        selection_mode='RWS', # SUS, RWS
        reproduction_mode='GENETIC_ALGORITHM',  # 'GENETIC_ALGORITHM' 'HILL_CLIMBING'
        mutation_variance=0.1,
        elitist_fraction=0.1,
        mating_fraction=0.9,
        crossover_probability=0.5,
        crossover_mode='1-POINT',
        max_generation=100,
        termination_function=None,
        checkpoint_interval=1,
        timeit=True
    )
    evo.run()
    evo.timing.report()


def test_continuation():

    eval_func = lambda pop,rand_seed: RandomState(rand_seed).random(len(pop))

    print('Loading evolution from json file')
    folder_path = './tmp_cont1'
    utils.make_dir_if_not_exists(folder_path)
    evo1 = Evolution(
        random_seed=123,
        population_size=1000,
        genotype_size=2,
        evaluation_function=eval_func,
        fitness_normalization_mode='RANK',
        selection_mode='RWS',
        reproduction_mode='GENETIC_ALGORITHM', #'HILL_CLIMBING',  'GENETIC_ALGORITHM'
        mutation_variance=0.1,
        elitist_fraction=0.1,
        mating_fraction=0.9,
        crossover_probability=0.5,
        crossover_mode='1-POINT',
        max_generation=100,
        folder_path=folder_path,
        termination_function=None,
        checkpoint_interval=50
    )
    evo1.run()

    print()

    new_folder_path = './tmp_cont2'
    utils.make_dir_if_not_exists(new_folder_path)
    evo2 = Evolution.load_from_file(
        os.path.join(folder_path, 'evo_00050.json'),
        evaluation_function=eval_func,
        folder_path=new_folder_path,
        max_generation=110
    )
    evo2.run()


if __name__ == "__main__":
    # test_evolution()
    test_continuation()
    # time_evolution()
    # rws_test()
