=========
Changelog
=========

Version 0.0.1
=============
- initial build of the package

Version 0.0.2
=============
- introduced 'NONE' option for 'fitness_normalization_mode' (use performances as fitnesses with no normalization)
- introduction 'UNIFORM' option for 'selection_mode' - in this case the mating pool is created by choosing genotype elements in sequence (as opposed to sampling).
- new option 'reproduce_from_elite' - to allow the mating_pool to be made of genotypes from elite population only

Version 0.0.3
=============
- introduced param 'performance_objective': 'MAX', 'MIN', 'ABS_MAX', float value

Version 0.1.0
=============
- updated min max rand int in utils

Version 0.2.0
=============
- keep track of unsorted population and sorted index for reproducibility

Version 0.3.0
=============
- single pop_eval_random_seed (no longer list of seeds)
- taking into rounding error for prob and cum prob tolerances
- fixes on checking parameters

Version 0.4.0
=============
- multi population (no backward compatible)
- adjusting rounding tolerances
- introduce check on perfromances (non-negative)
- put fitenss to None if fitness_norm_mode is NONE
- included population_unsorted in numpy conversion in load_from_file

Version 0.4.1
=============
- fix linear_scaling: error occured when all agents have same performance
- changes in requirments.txt (after pip install wheel)

Version 0.4.2
=============
- shuffle_agents param
