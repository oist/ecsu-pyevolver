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
- changed 'NONE' option for 'fitness_normalization_mode' with 'PERFORMANCE_MAX'
- adding similar related 'modes': 'PERFORMANCE_MIN', 'PERFORMANCE_ZERO', 'PERFORMANCE_ABS_MAX', 
