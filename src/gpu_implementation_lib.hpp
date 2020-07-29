#ifndef gpu_implementation_lib_hpp
#define gpu_implementation_lib_hpp

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cmath>


#include "../config.h"
#include "structs_lib.cpp"

using namespace std;

__host__ __device__ bool operator<(const Member &, const Member &);
__global__ void Generate_Population_gpu(Member *, float *, float *);
__global__ void Count_Fitness_gpu(Member*);
__global__ void Count_Probability(Member*);
__global__ void Crossover_gpu(Member *, Member *, float *, float *,float *, float *, float *);
__global__ void Mutation_gpu(Member *, float *, float *, float *, float *);
bool compareByFit_gpu(const Member &, const Member &);

#endif
