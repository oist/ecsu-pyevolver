"""
TODO: Missing module docstring
"""
from pyevolver import evolution
from pyevolver.evolution import Evolution, MIN_SEARCH_VALUE, MAX_SEARCH_VALUE
from pyevolver import utils
import numpy as np
from pytictoc import TicToc
from numpy.random import RandomState
import os


#################################
# START OF ASSERTION FUNCTIONS
#################################

def test_linear_scaling():

    performances = np.array([1, 50, 15, 21, 100, 23, 88, 45, 44, 76])

    avg_perf = np.mean(performances)
    max_perf = np.max(performances)
    min_perf = np.min(performances)
    print('avg_perf: {}'.format(avg_perf))
    print('max_perf: {}'.format(max_perf))
    print('min_perf: {}'.format(min_perf))

    m = utils.linear_scaling(min_perf, max_perf, avg_perf, evolution.DEFAULT_MAX_EXPECTED_OFFSPRING)
    # print('m: {}'.format(m))
    assert m == 0.08621973929236505
    print('SUCCESS!')


def test_fitness_FPS():

    performances = np.array([1, 50, 15, 21, 100, 23, 88, 45, 44, 76])

    evo = Evolution(
        population_size=10,
        genotype_size=0,
        fitness_normalization_mode='FPS',
        evaluation_function=lambda pop,seeds: performances,
        max_generation=0
    )

    evo.run()

    correct_fitness_fps = sorted(
        [
            0.09156424581005587, 0.10068901303538176, 0.09417132216014897,
            0.09528864059590317, 0.11000000000000001, 0.0956610800744879,
            0.10776536312849162, 0.09975791433891994, 0.09957169459962756, 0.10553072625698326
        ], reverse=True
    )
    assert list(evo.fitnesses) == correct_fitness_fps
    # print('fitnesses_fps: {}'.format(fitnesses_fps))
    print('SUCCESS!')


def test_fitness_RANK():

    performances = np.array([1, 50, 15, 21, 100, 23, 88, 45, 44, 76])

    evo = Evolution(
        population_size=10,
        genotype_size=0,
        fitness_normalization_mode='RANK',
        evaluation_function=lambda pop, seeds: performances,
        max_generation=0
    )

    evo.run()

    correct_fitness_rank = sorted(
        [
            0.11000000000000001, 0.10777777777777779, 0.10555555555555556,
            0.10333333333333335, 0.10111111111111111, 0.09888888888888889,
            0.09666666666666666, 0.09444444444444444, 0.09222222222222222, 0.09
        ], reverse=True
    )
    assert list(evo.fitnesses) == correct_fitness_rank
    # print('fitnesses_rank: {}'.format(fitnesses_rank))
    print('SUCCESS!')


def test_crossover():

    evo = Evolution(
        population_size=0,
        genotype_size=10,
        evaluation_function=lambda pop, seeds: 0,
        # crossover_mode = "UNIFORM"
        crossover_mode="1-POINT",
        crossover_points=[2, 5, 8]
    )

    genotype1 = [0] * evo.genotype_size
    genotype2 = [1] * evo.genotype_size

    a, b = evo.crossover(genotype1, genotype2)
    print(''.join(str(int(x)) for x in a))
    print(''.join(str(int(x)) for x in b))


def test_select_mating_pool_RWS():

    from collections import Counter
    performances = np.array([1000, 1000, 300, 200, 200, 100, 50, 50, 50, 50])
    np.random.seed(11)

    evo = Evolution(
        population_size=10,
        population=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        genotype_size=1,
        fitness_normalization_mode='RANK',
        selection_mode='RWS',
        evaluation_function=lambda pop, seeds: performances,
        max_generation=0,
        elitist_fraction=0.2,
        mating_fraction=0.8,  # in beer this is 1 - elitist_fraction
    )

    evo.run()

    print("Fitnesses: {}".format(evo.fitnesses))

    # assert evo.n_elite == 2 # only in genetic algorithm
    # assert evo.n_fillup == 0  # only in genetic algorithm
    assert evo.n_mating == evo.population_size

    mating_counter = Counter()

    repetitions = 10000
    for _ in range(repetitions):
        mating_pool = evo.select_mating_pool()
        for e in mating_pool:
            mating_counter[e] += 1

    # we check if the number of time each genome is selected is proportionate to the its performance (and fitness)
    # not guaranteed to work for fitness_normalization_mode == 'FPS' because of the linear scaling
    # not easy to test this further (not implemented in beer)
    for i in range(2, 9):
        assert mating_counter[i] > mating_counter[i+1]

    print(sorted(mating_counter.items(), key=lambda kv: kv[0]))

    print('SUCCESS!')


def test_select_mating_pool_SUS():
    np.random.seed(11)

    evo = Evolution(
        population_size=10,
        population=np.array(list(range(1, 11))),
        genotype_size=1,
        fitness_normalization_mode='FPS',
        selection_mode='SUS',
        reproduction_mode='GENETIC_ALGORITHM',
        evaluation_function=lambda pop, seeds: [0.3, 0.2, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        max_generation=0,
        elitist_fraction=0.2,
        mating_fraction=0.8,  # in beer this is 1 - elitist_fraction
    )

    evo.run()
    evo.fitnesses = np.array([0.3, 0.2, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])

    # print("Fitnesses: {}".format(evo.fitnesses))

    assert evo.n_mating == 8
    assert evo.n_elite == 2
    assert evo.n_fillup == 0

    mating_pool = evo.select_mating_pool()

    print(mating_pool)
    assert mating_pool == [1, 1, 2, 2, 3, 5, 7, 10]

    print('SUCCESS!')


def test_mutate():

    np.random.seed(11)

    evo = Evolution(
        population_size=2,
        genotype_size=10,
        search_constraint=np.array([True] * 10),
        # search_constraint = np.array([False] * 10),
        evaluation_function=lambda pop, seeds: [1],
        max_generation=0,
    )
    genotype = np.array([-1]*10)
    mutant = evo.mutate(genotype)
    print(mutant)

    print('SUCCESS!')


def assertion_tests():
    test_linear_scaling()
    test_fitness_FPS()
    test_fitness_RANK()
    test_crossover()
    test_select_mating_pool_RWS()
    test_select_mating_pool_SUS()
    test_mutate()

#################################
# END OF ASSERTION FUNCTIONS
#################################


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


def time_evolution():
    evo = Evolution(
        random_seed=123,
        population_size=5000,
        genotype_size=100,
        evaluation_function=lambda pop, seeds: [1]*len(pop),
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

def test_v002():

    folder_path = './data/tmp1'
    utils.make_dir_if_not_exists(folder_path)

    evo = Evolution(
        random_seed=123,
        population_size=4,
        genotype_size=2,
        evaluation_function=lambda pop, seeds: np.sum(pop, axis=1), #np.arange(1,0,-1/len(pop)),
        fitness_normalization_mode='NONE',
        selection_mode='UNIFORM', # UNIFORM, SUS, RWS
        reproduce_from_elite=True,
        reproduction_mode='GENETIC_ALGORITHM',  # 'GENETIC_ALGORITHM' 'HILL_CLIMBING'
        mutation_variance=0.1,
        elitist_fraction=0.5,
        mating_fraction=0.5,
        crossover_probability=0.5,
        crossover_mode='UNIFORM',
        max_generation=100,
        termination_function=None,
        checkpoint_interval=1,
        folder_path=folder_path,
        timeit=True
    )
    evo.run()
    evo.timing.report()

def test_v003():

    folder_path = './data/tmp1'
    utils.make_dir_if_not_exists(folder_path)

    evo = Evolution(
        random_seed=np.random.randint(100000),
        population_size=4,
        genotype_size=2,
        evaluation_function=lambda pop, seeds: np.sum(pop, axis=1), #np.arange(1,0,-1/len(pop)),
        performance_objective=0.5, #MAX MIN ZERO ABS_MAX
        fitness_normalization_mode='NONE',
        selection_mode='UNIFORM', # UNIFORM, SUS, RWS
        reproduce_from_elite=True,
        reproduction_mode='GENETIC_ALGORITHM',  # 'GENETIC_ALGORITHM' 'HILL_CLIMBING'
        mutation_variance=0.1,
        elitist_fraction=0.5,
        mating_fraction=0.5,
        crossover_probability=0.5,
        crossover_mode='UNIFORM',
        max_generation=100,
        termination_function=None,
        checkpoint_interval=1,
        folder_path=folder_path,
        timeit=True
    )
    evo.run()
    evo.timing.report()

if __name__ == "__main__":
    # assertion_tests()
    # rws_test()
    # time_evolution()
    # test_continuation()
    # test_v002()
    test_v003()
    
    
