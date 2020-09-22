import os
import re
import json
from copy import deepcopy
from typing import List, Callable, Union
from dataclasses import dataclass, field, asdict
from itertools import cycle
import numpy as np
from pyevolver import utils
from pyevolver import json_numpy
from pyevolver.timing import Timing
from numpy.random import RandomState

np.seterr(over='ignore')

# GLOBAL DEFAULT PARAMS
FILE_NUM_ZFILL_DEFAULT = 5  # number of zeros for generation number in output file (e.g., 00001)
DEFAULT_MUTATION_VARIANCE = 0.1
DEFAULT_MAX_EXPECTED_OFFSPRING = 1.1
DEFAULT_CHECKPOINT_INTERVAL = 1  # number of generations after which to save the Evolution state
MIN_SEARCH_VALUE = -1  # min range of each genotype site
MAX_SEARCH_VALUE = 1  # max range of each genotype site
DEFAULT_CROSSOVER_PROB = 0.5  # default crossover probability


@dataclass
class Evolution:
    # pylint: disable=too-many-instance-attributes
    # noinspection PyUnresolvedReferences
    """
    Class that executes genetic search.
    :param population_size: (int) size of the population
    :param genotype_size: (int) size of the genotype vector
    :param evaluation_function: (func) function to evaluate genotype performance.
        It should take as inputthe entire population genotype (matrix) and return
        an array with the performances
    :param fitness_normalization_mode: (str) method to normalize fitness values
        (fitness-proportionate, rank-based or sigma scaling)
    :param selection_mode: (str) method to select parents for reproduction (RWS or SUS)
    :param reproduce_from_elite: (bool) whether the reproduction comes from elite 
        or remaining agents in the population
    :param reproduction_mode: (str) method to reproduce genetic algorithm or hill climbing
    :param mutation_variance: (float) variance of gaussian mutation rate
    :param folder_path: (string) path of the folder where to save the checkpoints
    :param search_constraint: (list of bool) flag whether to clip a specific site in
        a genotype (default to all True)
    :param reevaluate: (bool) whether to re-evaluate the individual if it's retained
        in the new generation
    :param max_generation: (int) maximum generations to evolve (not used if
        termination_function is provided)
    :param termination_function: (func) function to check if search should terminate
        (it accept the Evolution instance, default to None)
    :param elitist_fraction: (float) proportion of new population that will be made of
        best unmodified parents (only relevant for genetic algorithm)
    :param mating_fraction: (float) proportion of population that will be made of children
        (in Beer this is equal to 1. - elitist_fraction) (only relevant for genetic algorithm)
    :param crossover_probability: (float) probability that crossover will occur
        (only relevant for genetic algorithm)
    :param crossover_mode: (str) the way to perform crossover (UNIFORM, 1-POINT, 2-POINT, ...)
        (only relevant for genetic algorithm)
    :param crossover_points: (list of int) a list that specifies the indices of where
        to cut during crossover (only relevant for genetic algorithm)
    :param checkpoint_interval: (int) every how many generations should the population
        be saved and results logged
    :param max_expected_offspring: (float) number of offspring to be allocated to the
        best individual, best between 1 and 2
    """

    population_size: int
    genotype_size: int
    evaluation_function: Callable
    performance_objective: Union[str,float] = 'MAX' # 'MIN', 'ABS_MAX', float value
    fitness_normalization_mode: str = 'FPS' # 'NONE', 'FPS', 'RANK', 'SIGMA'
    selection_mode: str = 'RWS' # 'UNIFORM', 'RWS', 'SUS'
    reproduce_from_elite: bool = False
    reproduction_mode: str = 'HILL_CLIMBING' # 'HILL_CLIMBING', 'GENETIC_ALGORITHM'
    mutation_variance: float = DEFAULT_MUTATION_VARIANCE
    max_generation: int = 100
    termination_function: Callable = None
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL
    crossover_probability: float = DEFAULT_CROSSOVER_PROB
    crossover_points: List[int] = None
    folder_path: str = None
    elitist_fraction: float = None
    mating_fraction: float = None
    n_elite: int = None
    n_mating: int = None
    n_fillup: int = None
    crossover_mode: str = 'UNIFORM'
    search_constraint: np.ndarray = None  # this will be converted to all True by default in __post_init__
    reevaluate: bool = True
    max_expected_offspring: float = DEFAULT_MAX_EXPECTED_OFFSPRING

    random_seed: int = 0
    random_state: RandomState = None
    pop_eval_random_seed: np.ndarray = None  # initialized at every generation

    # other field (no need to define them outside)
    generation: int = 0  # the current generation number
    population: np.ndarray = None  # the list of population genotypes
    # (will be initialized in __post_init__)
    performances: np.ndarray = None  # performances of the genotypes
    fitnesses: np.ndarray = None  # fitnesses of the genotypes

    # collect average, best and worst performances across generations
    avg_performances: List[float] = field(default_factory=list)
    best_performances: List[float] = field(default_factory=list)
    worst_performances: List[float] = field(default_factory=list)

    timeit: bool = False

    def __post_init__(self):

        assert self.population_size % 2 == 0, "Population size must be even"

        if self.random_state is None:
            self.random_state = RandomState(self.random_seed)

        self.loaded_from_file = all(
            x is not None for x in 
            [self.population, self.performances, self.fitnesses]
        )

        # create initial population if not provided
        if self.population is None:
            # create a set of random genotypes
            self.population = self.random_state.uniform(
                MIN_SEARCH_VALUE, MAX_SEARCH_VALUE,
                [self.population_size, self.genotype_size]
            )

        if self.search_constraint is None:
            self.search_constraint = np.array([True] * self.genotype_size)

        self.file_num_zfill = int(np.ceil(np.log10(self.max_generation + 1))) \
            if self.max_generation \
            else 1 if self.max_generation == 0 \
            else FILE_NUM_ZFILL_DEFAULT

        # conver performance_objective to float if it is a string with a number
        f = utils.get_float(self.performance_objective)
        if f is not None:
            self.performance_objective = f

        self.timing = Timing(self.timeit)

        self.validate_params()
        self.init_reproduction_parameters()

    @staticmethod
    def get_random_genotype(rando_state, gen_size):
        return rando_state.uniform(MIN_SEARCH_VALUE, MAX_SEARCH_VALUE, gen_size)

    def init_reproduction_parameters(self):
        # self.n_mating: number of new agents return by select_mating_pool()
        if self.reproduction_mode == 'GENETIC_ALGORITHM':
            # self.n_elite: number of best agents to preserve (only used in genetic algorithm)
            # self.n_fillup: agents to be randomly generated
            self.n_elite = int(
                np.floor(self.population_size * self.elitist_fraction + 0.5)
            )  # children from elite group
            self.n_mating = int(np.floor(
                self.population_size * self.mating_fraction + 0.5 # at least one
            ))  # children from mating population
            self.n_fillup = self.population_size - (self.n_elite + self.n_mating)  # children from random fillup
            assert self.n_elite + self.n_mating + self.n_fillup == self.population_size
        else:  # 'HILL_CLIMBING'
            self.n_mating = self.population_size

    def validate_params(self):

        # termination condition
        assert self.max_generation is None or self.termination_function is None, \
            "Either max_generation or termination_function must be defined"

        # folder path
        if self.folder_path:
            assert os.path.isdir(self.folder_path), "folder_path '{}' is not a valid directory".format(self.folder_path)

        # search_constraint
        assert len(self.search_constraint) == self.genotype_size, \
            "The length of search_constraint should be equal to genotype_size"

        # performance_objective         
        accepted_values = ['MAX', 'MIN', 'ABS_MAX']
        assert type(self.performance_objective) in [float,int] or \
            self.performance_objective in accepted_values, \
            'performance_objective should be either {}'.format(', '.join(accepted_values))

        # fitness_normalization_mode         
        accepted_values = ['NONE', 'FPS', 'RANK', 'SIGMA']
        assert self.fitness_normalization_mode in accepted_values, \
            'fitness_normalization_mode should be either {}'.format(', '.join(accepted_values))
        assert self.fitness_normalization_mode.startswith!='NONE' or self.selection_mode == 'UNIFORM', \
            "if fitness_normalization_mode is 'NONE' (copy of PERFORMANCE), selection_mode must be UNIFORM (not normalized)" 

        # selection_mode
        accepted_values = ['UNIFORM', 'RWS', 'SUS']
        assert self.selection_mode in accepted_values, \
            'selection_mode should be either {}'.format(', '.join(accepted_values))

        # reproduction_mode
        accepted_values = ['HILL_CLIMBING', 'GENETIC_ALGORITHM']
        assert self.reproduction_mode in accepted_values, \
            'reproduction_mode should be either {}'.format(', '.join(accepted_values))

        # GENETIC_ALGORITHM
        if self.reproduction_mode == 'GENETIC_ALGORITHM':
            assert 0 <= self.elitist_fraction <= 1, \
                'In GENETIC_ALGORITHM: 0 <= elitist_fraction <=1'
            assert 0 <= self.mating_fraction <= 1, \
                'In GENETIC_ALGORITHM: 0 <= mating_fraction <=1'
            assert 0 <= self.crossover_probability <= 1, \
                'In GENETIC_ALGORITHM: 0 <= crossover_probability <=1'
            assert re.match('UNIFORM|\d+-POINT', self.crossover_mode), \
                'In GENETIC_ALGORITHM: crossover_mode should be UNIFORM or x-POINT'

        # crossover
        assert self.crossover_mode != None, "crossover_mode cannot be None"        
        if self.crossover_mode == 'UNIFORM':
            # crossover is computed on the entire genotype
            # with prob 0.5 of flipping each genotype site
            assert self.crossover_points is None, \
                "In uniform crossover_mode you shouldn't specify the crossover_points"
        elif self.crossover_mode.endswith('-POINT'):
            # A. if crossover_points is None the points are randomly generated
            # crossover_points must be a list of max x-1 integers in the interval [1,G-1]
            # where x is the integer > 0 specified in the parameter crossover_mode ('x-POINT')
            # and G is the size of the genotype
            # e.g. if parent1=[0,0,0] and parent2=[1,1,1] (G=3),
            # crossover_points must contain a single integer which can be
            # 1: child1=[0,1,1] child2=[1,0,0]
            # 2: child1=[0,0,1] child2=[1,1,0]
            # B. if crossover_points is not None -> num_points <= len(self.crossover_points)
            # if num_points < len(self.crossover_points)
            # only num_points will be randomly selected from the self.crossover_points
            num_points = self.crossover_mode[:-6]
            assert utils.is_int(num_points), \
                "Param crossover_mode should be 'UNIFORM' or 'x-POINT' (with x being an integer > 0)"
            num_points = int(num_points)
            assert 0 < num_points < self.genotype_size, \
                "Param crossover_mode should be 'x-POINT', with x being an integer such that 0 < x < G " \
                "and where G is the size of the genotype"
            assert num_points <= self.genotype_size - 1, \
                "Too high value for {} in param crossover_mode. Max should be G-1 " \
                "(where G is the size of the genotype)".format(
                    self.crossover_mode)
            if self.crossover_points is not None:
                assert len(set(self.crossover_points)) == len(self.crossover_points), \
                    "Duplicated values in crossover_points"
                self.crossover_points = sorted(set(self.crossover_points))
                assert num_points <= len(self.crossover_points), \
                    "crossover_mode={} and crossover_points={} but {} must be <= {}=len(crossover_points)".format(
                        self.crossover_mode, self.crossover_points, num_points, len(self.crossover_points))
                assert all(1 < x < self.genotype_size for x in self.crossover_points), \
                    "Some of the values in crossover_points are not in the interval [1,G-1] " \
                    "where G is the size of the genotype"
        else:
            assert False, \
                "Param crossover_mode should be 'UNIFORM' or 'x-POINT' (with x being an integer > 0)"

    def set_folder_name(self, text):
        self.folder_path = text


    def run(self):
        """
        Execute a full search run until some condition is reached.
        :return: the last population in the search
        """

        if self.loaded_from_file:
            # comple cycle from previous run (after saving)
            self.save_to_file()
            self.reproduce()
            self.generation += 1

        t = self.timing.init_tictoc()

        while self.generation <= self.max_generation:
            # evaluate all genotypes on the task
            self.pop_eval_random_seed = self.random_state.randint(0, 2 ** 32, self.population_size)

            self.performances = self.evaluation_function(self.population, self.pop_eval_random_seed)                        
            if type(self.performances) != np.ndarray:
                self.performances = np.array(self.performances)

            assert len(self.performances) == self.population_size

            self.timing.add_time('EVO1-RUN_eval_function', t)

            self.sort_population_on_performance()

            self.timing.add_time('EVO1-RUN_sort_population', t)

            # update average/best/worst population performance
            avg, best, worst = np.mean(self.performances), self.performances[0], self.performances[-1]
            variance = np.var(self.performances)
            self.avg_performances.append(avg)
            self.best_performances.append(best)
            self.worst_performances.append(worst)
            self.timing.add_time('EVO1-RUN_stats', t)

            # Compute fitnesses (based on performances)
            self.update_fitnesses()
            self.timing.add_time('EVO1-RUN_update_fitness', t)

            # print short statistics
            print("Generation {}: Best: {:.5f}, Worst: {:.5f}, Average: {:.5f}, Variance: {:.5f}".format(
                str(self.generation).rjust(self.file_num_zfill), best, worst, avg, variance))
            self.timing.add_time('EVO1-RUN_print_stats', t)

            # check if to terminate
            if self.generation == self.max_generation or \
                    (self.termination_function and self.termination_function(self)):
                self.save_to_file()
                # Stop search due to termination condition
                break

            # save the intermediate evolution state
            if self.checkpoint_interval and self.generation % self.checkpoint_interval == 0:
                # save current generation
                self.save_to_file()
            self.timing.add_time('EVO1-RUN_savefile', t)

            # run reproduce
            self.reproduce()
            self.timing.add_time('EVO1-RUN_reproduce', t)

            # update generation
            self.generation += 1

    def sort_population_on_performance(self):
        
        if type(self.performance_objective) is str:
            if self.performance_objective == 'MAX':            
                performances_objectified = self.performances
            elif self.performance_objective == 'MIN':
                performances_objectified = - self.performances
            elif self.performance_objective == 'ABS_MAX':
                performances_objectified = np.abs(self.performances)
        else:
            # minimizing the distance between performance and perf objective
            # when self.performance_objective==0 this would be identical to 'ABS_MIN'
            performances_objectified = - np.abs(self.performances - self.performance_objective)

        # sort genotypes and performances by performance_objectified from hight to low
        perf_sort_indexes = np.argsort(-performances_objectified)            
        self.performances = np.take_along_axis(self.performances, perf_sort_indexes, axis=None)
        perf_sort_indexes = perf_sort_indexes.reshape(self.population_size, 1)
        self.population = np.take_along_axis(self.population, perf_sort_indexes, axis=0)

        # OLD METHOD WITHOUT NUMPY:
        # sort genotypes and performances by performance from best to worst
        # self.population, self.performances = \
        #     zip(*sorted(zip(self.population, self.performances), 
        #     key=lambda pair: pair[1], reverse=True))
        # self.population = np.array(self.population)
        # self.performances = np.array(self.performances)


    def reproduce(self):
        """Run reproduce via HILL_CLIMBING or GENETIC_ALGORITHM"""
        if self.reproduction_mode == 'GENETIC_ALGORITHM':
            self.reproduce_genetic_algorithm()
        else:
            self.reproduce_hill_climbing()

    def reproduce_genetic_algorithm(self):
        """
        Reproduce a single generation in the following way:
        1) Copy the proportion equal to elitist_fraction of the current population to the new population
           (these are best_genotypes)
        2) Select part of the population for crossover using some selection method (set in config)
        3) Shuffle the selected population in preparation for cross-over
        4) Create crossover_fraction children of selected population with probability of crossover equal
           to prob_crossover.
        Crossover takes place at genome module boundaries (single neurons).
        5) Apply mutation to the children with mutation equal to mutation_var
        6) Fill the rest of the population with randomly created genotypes

        self.population and self.performances are sorted based on performances
        """

        t = self.timing.init_tictoc()

        new_population = [None] * self.population_size

        # 1) Elitist selection
        self.elite_population = self.population[:self.n_elite]
        new_population[:self.n_elite] = deepcopy(self.elite_population)
        self.timing.add_time('EVO2-GA_1_elitist_selection', t)

        # 2) Select mating population from the remaining population        
        mating_pool = self.select_mating_pool()
        self.timing.add_time('EVO2-GA_2_mating_pool', t)

        # 3) Shuffle
        self.random_state.shuffle(mating_pool)
        self.timing.add_time('EVO2-GA_3_shuffle', t)

        # 4) Create children with crossover or apply mutation
        mating_finish = self.n_elite + self.n_mating
        newpop_counter = self.n_elite  # track where we are in the new population
        mating_counter = 0
        
        while newpop_counter < mating_finish:
            not_last = mating_finish - newpop_counter > 1
            parent1 = mating_pool[mating_counter]

            if not_last and self.random_state.random() < self.crossover_probability:
                parent2 = mating_pool[mating_counter + 1]
                child1, child2 = self.crossover(parent1, parent2)
                # if the child is the same as the first parent after crossover, mutate it (as in Beer)
                if np.array_equal(child1, parent1):
                    child1 = self.mutate(parent1)
                new_population[newpop_counter], new_population[newpop_counter + 1] = child1, child2
                newpop_counter += 2
                mating_counter += 2
            else:
                # if no crossover, mutate just one genotype
                child1 = self.mutate(parent1)
                new_population[newpop_counter] = child1
                newpop_counter += 1
                mating_counter += 1
        self.timing.add_time('EVO2-GA_4_children', t)

        # 5) Fill up with random new genotypes
        new_population[newpop_counter:] = self.random_state.uniform(0, 1, [self.n_fillup, self.genotype_size])
        self.timing.add_time('EVO2-GA_5_fillup', t)

        # 6) redefined population based on the newly computed population
        self.population = np.array(new_population)
        self.timing.add_time('EVO2-GA_6_convert_pop', t)

    def reproduce_hill_climbing(self):

        t = self.timing.init_tictoc()

        # 1) Select the parents using sampling (replacing the entire population, no elite here)
        parent_population = self.select_mating_pool()
        self.timing.add_time('EVO2-HC_1_mating pool', t)

        # 2) Reevaluate
        if self.reevaluate:
            parent_performance = np.array(self.evaluation_function(parent_population, self.pop_eval_random_seed))
        else:
            assert False, \
                "reevaluate params has to be True. " \
                "For reevaluate to be False we need to also return performances in function select_mating_pool"
        self.timing.add_time('EVO2-HC_2_reevaluate', t)

        # 3) Produce the new population by mutating each parent and rewrite it on the current population
        self.population = np.array([self.mutate(gen) for gen in parent_population])
        self.timing.add_time('EVO2-HC_3_mutate', t)

        # 4) Calculate new performances
        self.performance = np.array(self.evaluation_function(self.population, self.pop_eval_random_seed))
        self.timing.add_time('EVO2-HC_4_compute_perf', t)

        # 5) Check if performace worsened and in this case retrieve agent from parent population
        lower_performance = self.performance < parent_performance  # bool array
        for i in range(self.population_size):
            if lower_performance[i]:
                self.population[i] = parent_population[i]
                self.performance[i] = parent_performance[i]
        self.timing.add_time('EVO2-HC_5_compare_and_select', t)

    def update_fitnesses(self):
        """
        Update genotype fitness to relative values, retain sorting from best to worst.
        """        
        if self.fitness_normalization_mode == 'NONE':
            # use performances as fitnesses
            self.fitnesses = np.copy(self.performances)

        elif self.fitness_normalization_mode == 'FPS':  # (fitness-proportionate)
            avg_perf = self.avg_performances[-1]
            m = utils.linear_scaling(
                self.worst_performances[-1],
                self.best_performances[-1],
                avg_perf,
                self.max_expected_offspring
            )
            scaled_performances = m * (self.performances - avg_perf) + avg_perf
            total_performance = np.sum(scaled_performances)
            self.fitnesses = scaled_performances / total_performance

        elif self.fitness_normalization_mode == 'RANK':  # (rank-based)
            # Baker's linear ranking method: f(pos) = 2-SP+2*(SP-1)*(pos-1)/(n-1)
            # the highest ranked individual receives max_exp_offspring (typically 1.1),
            # the lowest receives 2 - max_exp_offspring
            # normalized to sum to 1
            self.fitnesses = np.array(
                [(self.max_expected_offspring + (2 - 2 * self.max_expected_offspring) * i /
                  (self.population_size - 1)) / self.population_size for i in range(self.population_size)])

        elif self.fitness_normalization_mode == 'SIGMA':  # (sigma-scaling)
            # for every individual 1 + (I(f) - P(avg_f))/2*P(std) is calculated
            # if value is below zero, a small positive constant is given so the individual has some probability
            # of being chosen. The numbers are then normalized
            avg = np.mean(self.performances).item()
            std = max(0.0001, np.std(self.performances).item())
            exp_values = list((1 + ((f - avg) / (2 * std))) for f in self.performances)

            for i, v in enumerate(exp_values):
                if v <= 0:
                    exp_values[i] = 1 / self.population_size
            s = sum(exp_values)
            self.fitnesses = np.array(list(e / s for e in exp_values))

    def select_mating_pool(self):
        """
        Select a mating pool population.
        :return: selected parents for reproduction
        """

        mating_pool = []

        source_population = \
            self.elite_population if self.reproduce_from_elite \
            else self.population

        assert len(source_population>0), "Error, can't create a mating pool from empty source population"

        if self.selection_mode == 'UNIFORM':
            # create mating_pool from source_population uniformally 
            # (from beginning to end and if needed restart from beginning)
            cycle_source_population = cycle(source_population)
            mating_pool = [next(cycle_source_population) for _ in range(self.n_mating)]
        else:
            cum_probs = np.cumsum(self.fitnesses)
            assert 0 <= abs(cum_probs[-1] - 1.0) < 1E-13
            if self.selection_mode == "RWS":
                # roulette wheel selection
                mating_pool_indexes = self.random_state.choice(
                    self.population_size, size=self.n_mating, replace=True, p=self.fitnesses)
                mating_pool = [source_population[i] for i in mating_pool_indexes]
            elif self.selection_mode == "SUS":
                # TODO: find a way to implement this via numpy
                # stochastic universal sampling selection
                p_dist = 1 / self.n_mating  # distance between the pointers
                start = self.random_state.uniform(0, p_dist)
                pointers = [start + i * p_dist for i in range(self.n_mating)]

                for p in pointers:
                    for (i, genotype) in enumerate(source_population):
                        if p <= cum_probs[i]:
                            mating_pool.append(genotype)
                            break
            else:
                assert False

        mating_pool = deepcopy(mating_pool)
        assert len(mating_pool) == self.n_mating
        return mating_pool

    def crossover(self, parent1, parent2):
        """
        Given two genotypes, create two new genotypes by exchanging their genetic material.
        :param parent1: first parent genotype
        :param parent2: second parent genotype
        :return: two new genotypes
        # TODO: implement class testing functions
        """

        genotype_size = len(parent1)
        if self.crossover_mode == 'UNIFORM':
            if self.crossover_points is None:
                # by default do crossover on the entire genotype
                flips = self.random_state.choice(a=[0, 1], size=genotype_size)
            else:
                # TODO: this will never occur because we check crossover_points above but
                # consider implementing in the future a case of uniform crossover in certain
                # portions of the genotype
                assert False
            inv_flips = 1 - flips
            child1 = flips * parent1 + inv_flips * parent2
            child2 = flips * parent2 + inv_flips * parent1
        else:
            # x-POINT
            num_points = int(self.crossover_mode[:-6])
            if self.crossover_points is None:
                possible_points = list(range(1, genotype_size))  # [1,...,G-1]
                chosen_crossover_points = sorted(self.random_state.choice(possible_points, num_points, replace=False))
            elif num_points < len(self.crossover_points):
                chosen_crossover_points = sorted(
                    self.random_state.choice(self.crossover_points, num_points, replace=False))
            else:
                chosen_crossover_points = sorted(self.crossover_points)
                assert num_points == len(chosen_crossover_points)
            gt = [parent1, parent2]
            boundaries = [0] + chosen_crossover_points + [genotype_size]
            segment_ranges = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
            segments1 = [gt[i % 2][s[0]:s[1]] for i, s in enumerate(segment_ranges)]
            segments2 = [gt[1 - i % 2][s[0]:s[1]] for i, s in enumerate(segment_ranges)]
            child1 = np.hstack(segments1)
            child2 = np.hstack(segments2)
        return child1, child2

    def mutate(self, genotype):
        magnitude = self.random_state.normal(0, np.sqrt(self.mutation_variance))
        unit_vector = utils.make_rand_vector(len(genotype), self.random_state)
        mutant = np.where(
            self.search_constraint,
            np.clip(
                genotype + magnitude * unit_vector,
                MIN_SEARCH_VALUE,
                MAX_SEARCH_VALUE
            ),
            genotype + magnitude * unit_vector
        )
        return mutant

    def save_to_file(self):
        if self.folder_path is None:
            return
            # population is saved after sorting based on fitness
        file_path = os.path.join(
            self.folder_path,
            'evo_{}.json'.format(str(self.generation).zfill(self.file_num_zfill))
        )

        # print("Saving rand state: {}".format(state_of_rand_state))

        obj_dict = asdict(self)
        del obj_dict['evaluation_function']
        del obj_dict['termination_function']
        obj_dict['random_state'] = json_numpy.dumps(self.random_state.get_state())

        with open(file_path, 'w') as f_out:
            json.dump(obj_dict, f_out, cls=json_numpy.NumpyListJsonEncoder, indent=3)

    @staticmethod
    def load_from_file(file_path, evaluation_function: Callable = None,
                       termination_function: Callable = None,
                       **kwargs):

        with open(file_path) as f_in:
            obj_dict = json.load(f_in)

        for k in ['population', 'performances', 'fitnesses', 'pop_eval_random_seed']:
            # assert type(obj_dict[k]) == np.ndarray
            obj_dict[k] = np.array(obj_dict[k])

        random_state = RandomState(None)
        random_state_state = json_numpy.loads(obj_dict['random_state'])
        # print("Loading rand state: {}".format(random_state_state))
        random_state.set_state(random_state_state)
        obj_dict['random_state'] = random_state

        obj_dict['evaluation_function'] = evaluation_function
        obj_dict['termination_function'] = termination_function

        if kwargs:
            obj_dict.update(kwargs)

        evo = Evolution(**obj_dict)

        return evo


class DefaultTerminationFunction:

    def __init__(self):
        self.max_num_equal_performances = 10

        # stop the search if performance hasn't increased in a set number of generations

    def termination_function(self, evolution_obj):
        # take the set of the tail part of the performance array and check if it is made of a single value
        return len(set(evolution_obj.best_performances[-self.max_num_equal_performances:])) == 1
