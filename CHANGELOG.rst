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
