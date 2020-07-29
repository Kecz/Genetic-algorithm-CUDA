#include "gpu_implementation_lib.hpp"


__host__ __device__ bool operator<(const Member &lhs, const Member &rhs)
{ return (lhs.fitness > rhs.fitness); };


__global__ void Generate_Population_gpu(Member *Population_gpu_dev, float *Random_x, float *Random_y)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < How_many_members)
	{
		Population_gpu_dev[index].x = Random_x[index];
		Population_gpu_dev[index].y = Random_y[index];
	}
}

__global__ void Count_Fitness_gpu(Member* Population_gpu_dev)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < How_many_members)
	{
		if (WHICH_FUNCTION == 1)
			Population_gpu_dev[index].fitness = -(sin(Population_gpu_dev[index].x) + cos(Population_gpu_dev[index].y));
		else if (WHICH_FUNCTION == 2)
			Population_gpu_dev[index].fitness = -((Population_gpu_dev[index].x)*(Population_gpu_dev[index].x) + (Population_gpu_dev[index].y)*(Population_gpu_dev[index].y));
		else if (WHICH_FUNCTION == 3)
			Population_gpu_dev[index].fitness = -((Population_gpu_dev[index].y)*sin(Population_gpu_dev[index].x) - (Population_gpu_dev[index].x)*cos(Population_gpu_dev[index].y));
		else if (WHICH_FUNCTION == 4)
			Population_gpu_dev[index].fitness = -(sin(Population_gpu_dev[index].x + Population_gpu_dev[index].y) + (Population_gpu_dev[index].x - Population_gpu_dev[index].y)*(Population_gpu_dev[index].x - Population_gpu_dev[index].y) - 1.5*Population_gpu_dev[index].x + 2.5*Population_gpu_dev[index].y + 1);
		else if (WHICH_FUNCTION == 5)
			Population_gpu_dev[index].fitness = -(-(Population_gpu_dev[index].y + 47)*sin(sqrt(abs((Population_gpu_dev[index].x / 2) + (Population_gpu_dev[index].y + 47)))) - Population_gpu_dev[index].x*sin(sqrt(abs(Population_gpu_dev[index].x - (Population_gpu_dev[index].y + 47)))));
		else if (WHICH_FUNCTION == 6)
			Population_gpu_dev[index].fitness = -(418.9829 * 2 - (Population_gpu_dev[index].x*sin(sqrt(abs(Population_gpu_dev[index].x))) + Population_gpu_dev[index].y*sin(sqrt(abs(Population_gpu_dev[index].y)))));

	}
}

__global__ void Count_Probability(Member* Population_gpu_dev)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < How_many_members)
	{
		if(index != (How_many_members-1))
			Population_gpu_dev[index].probability = RANK_STEP_DOWN * pow((1 - RANK_STEP_DOWN), index);
		else if (index == (How_many_members - 1))
			Population_gpu_dev[index].probability = pow((1 - RANK_STEP_DOWN), index);
	}
}

__global__ void Crossover_gpu(Member *Population, Member *Population_new, float *do_crossover, float *rand_cross_x,float * rand_cross_y, float * parent1_rand, float * parent2_rand)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < How_many_members)
	{
		int parent1 = 0;

		if (do_crossover[index] < CROSSOVER_PROBABILITY)
		{
			//Choosing parent 1
			float offset = 0.0;
			//Roulette selection
			for (int i = 0; i < How_many_members; i++)
			{
				offset += Population[i].probability;
				if (offset > parent1_rand[index])
				{
					parent1 = i;
					break;
				}
			}

			//Choosing parent 2
			offset = 0.0;
			int parent2 = 0;

			//Roulette selection
			for (int i = 0; i < How_many_members; i++)
			{
				offset += Population[i].probability;
				if (offset > parent2_rand[index])
				{
					parent2 = i;
					break;
				}
			}

			//Sorting parents descending - at the end parent1 has bigger value of fitness, and parent2 has smaller.
			if (Population[parent1].fitness < Population[parent2].fitness)
			{
				int temp = parent1;
				parent1 = parent2;
				parent2 = temp;
			}

			float x_min = Population[parent2].x - alpha * (Population[parent1].x - Population[parent2].x);
			float x_max = Population[parent2].x + alpha * (Population[parent1].x - Population[parent2].x);
			
			float cross_x = x_min + (rand_cross_x[index])*(x_max - x_min);

			if (cross_x > range_max_x)
				cross_x = range_max_x;
			if (cross_x < range_min_x)
				cross_x = range_min_x;

			float y_min = Population[parent2].y - alpha * (Population[parent1].y - Population[parent2].y);
			float y_max = Population[parent2].y + alpha * (Population[parent1].y - Population[parent2].y);
			
			float cross_y = y_min + (rand_cross_y[index])*(y_max - y_min);

			if (cross_y > range_max_y)
				cross_y = range_max_y;
			if (cross_y < range_min_y)
				cross_y = range_min_y;

			Population_new[index].x = cross_x;
			Population_new[index].y = cross_y;

		}
		else
		{
			Population_new[index] = Population[parent1];
		}
	}
}

__global__ void Mutation_gpu(Member * thrust_pointer_new, float * do_mutation, float * rand_cross_x, float * rand_cross_y, float * mutation_step_maximum)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < How_many_members)
	{
		if (do_mutation[index] < MUTATION_PROBABILITY)
		{
			float mutation_value_x = (-mutation_step_maximum[index] + (rand_cross_x[index]) * (mutation_step_maximum[index] - (-mutation_step_maximum[index])));
			//thrust_pointer_new[index].x = thrust_pointer_new[index].x + (-mutation_step_maximum[index] + (rand_cross_x[index])*(mutation_step_maximum[index] - (-mutation_step_maximum[index])));
			
			if (thrust_pointer_new[index].x + mutation_value_x > range_max_x)
			{
				thrust_pointer_new[index].x = range_max_x - (mutation_value_x - (range_max_x- thrust_pointer_new[index].x));
			}
			else if (thrust_pointer_new[index].x + mutation_value_x < range_min_x)
			{
				thrust_pointer_new[index].x = range_min_x - (mutation_value_x - (range_min_x - thrust_pointer_new[index].x));
			}
			else
			{
				thrust_pointer_new[index].x = thrust_pointer_new[index].x + mutation_value_x;
			}

			
			//thrust_pointer_new[index].y = thrust_pointer_new[index].y + (-mutation_step_maximum[index] + (rand_cross_y[index])*(mutation_step_maximum[index] - (-mutation_step_maximum[index])));
			float mutation_value_y = (-mutation_step_maximum[index] + (rand_cross_y[index]) * (mutation_step_maximum[index] - (-mutation_step_maximum[index])));

			if (thrust_pointer_new[index].y + mutation_value_y > range_max_y)
			{
				thrust_pointer_new[index].y = range_max_y - (mutation_value_y - (range_max_y - thrust_pointer_new[index].y));
			}
			else if (thrust_pointer_new[index].y + mutation_value_y < range_min_y)
			{
				thrust_pointer_new[index].y = range_min_y - (mutation_value_y - (range_min_y - thrust_pointer_new[index].y));
			}
			else
			{
				thrust_pointer_new[index].y = thrust_pointer_new[index].y + mutation_value_y;
			}


		}
	}


}

bool compareByFit_gpu(const Member &a, const Member &b)
{
	return a.fitness > b.fitness;
}
