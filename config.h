#ifndef config_h
#define config_h

//Parameters to adjust algorithm
#define range_min_x (float) -500
#define range_max_x (float) 500
#define range_min_y (float) -500
#define range_max_y (float) 500
#define WHICH_FUNCTION 6
#define Do_show_members_cpu 0
#define Do_show_members_gpu 0
#define Do_show_generations 1
#define Do_save_to_file 1
#define do_run_on_cpu 1
#define do_run_on_gpu 1
#define how_many_included_average 1

#define How_many_members 5000
#define How_many_generations 3
//#define Mutation_step_min (float)0.1
#define Mutation_step_max (float)1000
#define Mutation_annealing (float) 0.96
#define alpha (float)0.1
#define RANK_STEP_DOWN (float)0.3
#define CROSSOVER_PROBABILITY (float)0.7
#define MUTATION_PROBABILITY (float)0.05

#define BLOCKS_PER_KERNEL 1000
#define THREADS_PER_BLOCK 1024

#endif