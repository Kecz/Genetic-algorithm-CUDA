//Bartosz Bielinski - Genetic Algorithm

/*
	Description of algorithm:
	- First population is generated randomly
	- Fitness is estimated based on value of selected optimized function in point (x,y)
	- Members are sorted on basis of value of fitness and then each member is getting probability based on "rank based selection", which can be adjusted with parameters 'RANK_STEP_DOWN' and 'CROSSOVER_PROBABILITY'
	- Roulette selection algorithm is used to choose random members
	- To crossover process the "Blend Crossover" for Real-Coded Genetic Algorithms is used and it can be adjusted with parameter 'alpha' 
	- In case of mutation, the mutated member is moved along x and y axis by radom value from range  <-Mutation_step_max * Mutation_annealing ^ generation_number, Mutation_step_max * Mutation_annealing ^ generation_number>, parameter to adjust: 'MUTATION_PROBABILITY', 'Mutation_step_max', 'Mutation_annealing'
	- Optimum is calculated as an average from a few best members, the amount of members included in average can be adjusted with parameter 'how_many_included_average' (default: 1) 
	- Each generation is saved to csv file, for CPU to file 'algorithm_results_cpu.csv', and for GPU to 'algorithm_results_gpu.csv'.
*/
/*
	Build-in functions:
	1. f(x,y) = sin(x) + cos(x) - no global minimum - any range of searching
	2. f(x,y) = x^2 + y^2 - parabola in 3D, global minimum in (0,0), f(0,0) = 0 - any range of searching
	3. f(x,y) = y*sin(x) - x*sin(y) - minimum can be find visually with plot_data.m, any range of searching
	4. McCormick Function - f(x,y) =  sin(x+y) + (x-y)^2 -1.5x +2.5y +1, global minimum in range -1.5<=x<=4  -3<=y<=4 in point (-0.54719, -1.54719) and f = -1.9133
	5. Eggholder function - in range -512<=x,y<=512 there is global minimum in (512, 404.2319) and f = -959.6407
	6. Schwefel Function - in range -500<=x,y<=500 there is global minimum in (420.9687, 420.9687) and f = 0.
*/

#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <time.h>       /* time */
#include <algorithm>    // std::sort
#include <fstream>
#include <chrono>
#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "config.h"
#include "src/cpu_implementation_lib.hpp"
#include "src/gpu_implementation_lib.hpp"
#include "src/structs_lib.cpp"


using namespace std;
using namespace std::chrono;


int main(int argc, char* argv[])
{
	srand(time(NULL));

	cout << "Bartosz Bielinski - Genetic Algorithm" << endl;

	string save_file_name_cpu, save_file_name_gpu;
	if (argc == 1)
    {
        printf("\nUsing default names for files with results: \n");
        save_file_name_cpu = "algorithm_results_cpu.csv";
        save_file_name_gpu = "algorithm_results_gpu.csv";
        cout << "CPU results filename: " << save_file_name_cpu << endl;
        cout << "GPU results filename: " << save_file_name_gpu << endl;
    }
    else if (argc == 3)
    {
        printf("\nUsing names for files with results passed with command line arguments: \n");
        save_file_name_cpu = argv[1];
        save_file_name_gpu = argv[2];
        cout << "CPU results filename: " << save_file_name_cpu << endl;
        cout << "GPU results filename: " << save_file_name_gpu << endl;
    }
    else
    {
		cout << "\nWrong input command line arguments\n" << endl;
		return 1;
    }

	duration<double> duration_gpu;
	duration<double> duration_cpu;

	if (do_run_on_cpu == 1)
	{
		ofstream file;

		cout << "Processing on CPU" << endl;

		if (Do_save_to_file == 1)
		{
			file.open(save_file_name_cpu, std::ios::trunc);
			if (file.good() == true)
			{
				cout << "The file was accessed" << endl;
				file << How_many_members << "," << How_many_generations << "," << WHICH_FUNCTION << "\n";

			}
			else
			{
				cout << "Access to the file was forbidden!" << std::endl;
				PauseSystem();
			}
		}

		//==============================================================================================================================
		//
		//
		// Genetic Algorithm on CPU
		//
		//
		//==============================================================================================================================


		high_resolution_clock::time_point begin_time_cpu = high_resolution_clock::now();

		vector <Member> Population;
		vector <Member> Population_new;

		//Generating population
		Generate_Population(Population);

		cout << "Size of population is " << Population.size() << endl;
		cout << "Amount of generations is: " << How_many_generations << endl;

		if (Do_show_members_cpu == 1)
			Show_Population(Population);

		float current_mutation_step_max = Mutation_step_max;
		//Course of many generations of population
		for (int gen = 0; gen < How_many_generations - 1; gen++)
		{
			if (Do_show_generations == 1)
				cout << endl << "Generation: " << gen + 1 << endl << endl;;

			Count_Fitness(Population);

			if (Do_show_members_cpu == 1)
				Show_Population(Population);

			if (Do_save_to_file == 1)
			{
				for (int j = 0; j < Population.size()-1; j++)
				{
					file << Population[j].x << "," << Population[j].y << ",";
				}
				file << Population[Population.size()-1].x << "," << Population[Population.size()-1].y << "\n";
			}


			//Sorting memebers descending
			sort(Population.begin(), Population.end(), compareByFit);

			if (Do_show_members_cpu == 1)
			{
				cout << endl << "Sorted population:" << endl;
				Show_Population(Population);
			}

			//Giving members a probability based on rank in rank base selection 
			for (int j = 0; j < Population.size(); j++)
			{

				if (j != (Population.size() - 1))
					Population[j].probability = RANK_STEP_DOWN * pow((1 - RANK_STEP_DOWN), j);
				else if (j == (Population.size() - 1))
					Population[j].probability = pow((1 - RANK_STEP_DOWN), j);
			}

			if (Do_show_members_cpu == 1)
				Show_Population(Population);

			//Crossover for every member
			for (int i = 0; i < Population.size(); i++)
			{
				float do_crossover = ((float)rand()) / (float)RAND_MAX;
				float do_mutation = ((float)rand()) / (float)RAND_MAX;

				if (do_crossover < CROSSOVER_PROBABILITY)
				{
					int parent1 = Roulete_Selection(Population);
					int parent2 = Roulete_Selection(Population);

					//Sorting parents descending - at the end parent1 has bigger value of fitness, and parent2 has smaller.
					if (Population[parent1].fitness < Population[parent2].fitness)
					{
						int temp = parent1;
						parent1 = parent2;
						parent2 = temp;

					}

					float x_min = Population[parent2].x - alpha * (Population[parent1].x - Population[parent2].x);
					float x_max = Population[parent1].x + alpha * (Population[parent1].x - Population[parent2].x);
					float cross_x = RandomFloat(x_min, x_max);

					if (cross_x > range_max_x)
						cross_x = range_max_x;
					if (cross_x < range_min_x)
						cross_x = range_min_x;

					float y_min = Population[parent2].y - alpha * (Population[parent1].y - Population[parent2].y);
					float y_max = Population[parent1].y + alpha * (Population[parent1].y - Population[parent2].y);
					float cross_y = RandomFloat(y_min, y_max);

					if (cross_y > range_max_y)
						cross_y = range_max_y;
					if (cross_y < range_min_y)
						cross_y = range_min_y;

					Member child;
					child.x = cross_x;
					child.y = cross_y;


					//Mutation 
					if (do_mutation < MUTATION_PROBABILITY)
						Mutate(child, current_mutation_step_max);

					Population_new.push_back(child);

				}
				else
				{
					int which_member_unchanged = Roulete_Selection(Population);
					Member unchanged_member = Population[which_member_unchanged];

					if (do_mutation < MUTATION_PROBABILITY)

						Mutate(unchanged_member, current_mutation_step_max);

					Population_new.push_back(unchanged_member);

				}
			}

			//Saving new population
			Population = Population_new;
			Population_new.clear();

			// Mutation max step annealing 
			current_mutation_step_max = Mutation_step_max * pow(Mutation_annealing, gen);
		}

		//Course of last generation
		if (Do_show_generations == 1)
			cout << endl << "Generacja: " << How_many_generations << endl << endl;;

		Count_Fitness(Population);

		if (Do_show_members_cpu == 1)
			Show_Population(Population);

		if (Do_save_to_file == 1)
		{
			for (int j = 0; j < Population.size()-1; j++)
			{
				file << Population[j].x << "," << Population[j].y << ",";
			}
			file << Population[Population.size()-1].x << "," << Population[Population.size()-1].y << "\n";
		}


		//Finding minimum based on the first best members (amount of included members can be adjusted with parameter 'how_many_included_average)
		sort(Population.begin(), Population.end(), compareByFit);

		float x_opt = 0;
		float y_opt = 0;

		for (int i = 0; i < how_many_included_average; i++)
		{
			x_opt += Population[i].x;
			y_opt += Population[i].y;
		}

		x_opt /= how_many_included_average;
		y_opt /= how_many_included_average;

		cout << "Optimum in x = " << x_opt << " y = " << y_opt << endl;

		Member member_opt;
		member_opt.x = x_opt;
		member_opt.y = y_opt;
		member_opt.fitness = FitFunction(member_opt);

		cout << "Value of in optimum optimum: " << -member_opt.fitness << endl;

		if (Do_save_to_file == 1)
		{
			file << x_opt << "," << y_opt << "\n";
			file << range_min_x << "," << range_max_x << "\n" << range_min_y << "," << range_max_y << "\n";

			if (file.good() == true)
			{
				cout << "Data successfully saved to file" << endl;
			}
			else
			{
				cout << "Access to the file was forbidden!" << std::endl;
				PauseSystem();
			}

			file.close();
		}

		high_resolution_clock::time_point end_time_cpu = high_resolution_clock::now();

		duration_cpu = duration_cast<duration<double>>(end_time_cpu - begin_time_cpu);

		cout << "Processing time on CPU: " << duration_cpu.count() << endl << endl;

		PauseSystem();

	}

	//==============================================================================================================================
	//
	//
	// Genetic algorithm on GPU
	//
	//
	//==============================================================================================================================

	if (do_run_on_gpu == 1)
	{
		cout << "Processing on GPU" << endl;

		ofstream file_gpu;

		if (Do_save_to_file == 1)
		{
			file_gpu.open(save_file_name_gpu, std::ios::trunc);
			if (file_gpu.good() == true)
			{
				cout << "You have access to the file!" << endl;
				file_gpu << How_many_members << "," << How_many_generations << "," << WHICH_FUNCTION << "\n";

			}
			else
			{
				cout << "Access to the file was forbidden!" << std::endl;
				PauseSystem();
			}
		}

		high_resolution_clock::time_point begin_time_gpu = high_resolution_clock::now();

		Member *Population_gpu = new Member[How_many_members];
		Member *Population_gpu_dev;

		unsigned long long int How_many_blocks = How_many_members / THREADS_PER_BLOCK + 1;
		cout << "Amount of blocks need on GPU: " << How_many_blocks << endl << endl;

		//Show population
		if (Do_show_members_gpu == 1)
			Show_Population(Population_gpu);

		int size = How_many_members * sizeof(Member);
		cudaMalloc(&Population_gpu_dev, size);
		cudaMemcpy(Population_gpu_dev, Population_gpu, size, cudaMemcpyHostToDevice);

		//Generating random numbers needed in generating generations
		float *Random_member_x = new float[How_many_members];
		for (int i = 0; i < How_many_members; i++)
			Random_member_x[i] = RandomFloat(range_min_x, range_max_x);

		float *dev_Random_member_x;
		cudaMalloc(&dev_Random_member_x, How_many_members * sizeof(float));
		cudaMemcpy(dev_Random_member_x, Random_member_x, How_many_members * sizeof(float), cudaMemcpyHostToDevice);

		//Generating random numbers needed in generating generations
		float *Random_member_y = new float[How_many_members];
		for (int i = 0; i < How_many_members; i++)
			Random_member_y[i] = RandomFloat(range_min_x, range_max_x);

		float *dev_Random_member_y;
		cudaMalloc(&dev_Random_member_y, How_many_members * sizeof(float));
		cudaMemcpy(dev_Random_member_y, Random_member_y, How_many_members * sizeof(float), cudaMemcpyHostToDevice);

		//Generating population
		Generate_Population_gpu <<<How_many_blocks, THREADS_PER_BLOCK >>> (Population_gpu_dev, dev_Random_member_x, dev_Random_member_y);

		cudaMemcpy(Population_gpu, Population_gpu_dev, size, cudaMemcpyDeviceToHost);

		cudaFree(dev_Random_member_x);
		cudaFree(dev_Random_member_y);

		delete[]Random_member_x;
		delete[]Random_member_y;

		if (Do_show_generations == 1)
			cout << endl << "Generation: 1" << endl << endl;

		if (Do_show_members_gpu == 1)
			Show_Population(Population_gpu);

		//Creating thrust vector to store population on host and device
		thrust::host_vector<Member> host_thrust_member(How_many_members);
		thrust::device_vector<Member> device_thrust_member(host_thrust_member);
		Member* thrust_pointer = thrust::raw_pointer_cast(&device_thrust_member[0]);	//Pointer on the beggining of device memory where population is stored


		for (int i = 0; i < How_many_members; i++)
		{
			host_thrust_member[i] = Population_gpu[i];
		}

		if (Do_save_to_file == 1)
		{
			for (int i = 0; i < How_many_members-1; i++)
			{
				file_gpu << host_thrust_member[i].x << "," << host_thrust_member[i].y << ",";
			}
			file_gpu << host_thrust_member[How_many_members-1].x << "," << host_thrust_member[How_many_members-1].y << "\n";
		}

		//Vectors in which are stored random numbers needed in process of crossover
		thrust::host_vector<float> do_crossover_gpu(How_many_members);
		thrust::host_vector<float> cross_x_gpu(How_many_members);
		thrust::host_vector<float> cross_y_gpu(How_many_members);
		thrust::host_vector<float> parent1_rand(How_many_members);
		thrust::host_vector<float> parent2_rand(How_many_members);
		thrust::host_vector<float> mutation_max_step_host(How_many_members);	//current mutation step

		for (int i = 0; i < How_many_members; i++)
		{
			mutation_max_step_host[i] = Mutation_step_max;
		}

		//Vectors in which are stored random numbers needed in process of mutation
		thrust::host_vector<float> do_mutation_gpu(How_many_members);
		thrust::host_vector<float> mutation_step_x_gpu(How_many_members);
		thrust::host_vector<float> mutation_step_y_gpu(How_many_members);


		//Course of many generations of population
		for (int gen = 0; gen < How_many_generations - 1; gen++)
		{
			if (Do_show_generations == 1)
				cout << endl << "Generation: " << gen + +2 << endl << endl;

			//Copying members from host to device
			device_thrust_member = host_thrust_member;

			if (Do_show_members_gpu == 1)
			{
				cout << "Preprocessed members" << endl;
				Member* Pointer_to_show_members = thrust::raw_pointer_cast(&host_thrust_member[0]);
				Show_Population(Pointer_to_show_members);
			}

			//Counting fitnesss for every member in population 
			Count_Fitness_gpu << < How_many_blocks, THREADS_PER_BLOCK >> > (thrust_pointer);

			if (Do_show_members_gpu == 1)
			{
				host_thrust_member = device_thrust_member;			//Copying members from device to host

				cout << "Processed memebers" << endl;
				Member* Pointer_to_show_members = thrust::raw_pointer_cast(&host_thrust_member[0]);
				Show_Population(Pointer_to_show_members);
			}


			//Sorting members in population descending based on fitness
			thrust::sort(device_thrust_member.begin(), device_thrust_member.end());

			if (Do_show_members_gpu == 1)
			{
				host_thrust_member = device_thrust_member;

				cout << endl << "Sorted population: " << endl;
				Member* Pointer_to_show_members = thrust::raw_pointer_cast(&host_thrust_member[0]);
				Show_Population(Pointer_to_show_members);
			}


			//Giving members a probability based on rank in rank base selection 
			Count_Probability << < How_many_blocks, THREADS_PER_BLOCK >> > (thrust_pointer);

			host_thrust_member = device_thrust_member;

			if (Do_show_members_gpu == 1)
			{
				cout << "With counted probablility" << endl;
				Member* Pointer_to_show_members = thrust::raw_pointer_cast(&host_thrust_member[0]);
				Show_Population(Pointer_to_show_members);
			}


			//Crossover

			//Vectors on device in which new population after crossover will be stored 
			thrust::device_vector<Member> device_thrust_member_new(host_thrust_member);
			Member* thrust_pointer_new = thrust::raw_pointer_cast(&device_thrust_member_new[0]);

			//Generating random numbers needed in kernels
			for (int i = 0; i < How_many_members; i++)
			{
				do_crossover_gpu[i] = ((float)rand()) / (float)RAND_MAX;
				if (do_crossover_gpu[i] < CROSSOVER_PROBABILITY)
				{
					cross_x_gpu[i] = ((float)rand()) / (float)RAND_MAX;
					cross_y_gpu[i] = ((float)rand()) / (float)RAND_MAX;
					parent1_rand[i] = ((float)rand()) / (float)RAND_MAX;
					parent2_rand[i] = ((float)rand()) / (float)RAND_MAX;
				}

				do_mutation_gpu[i] = ((float)rand()) / (float)RAND_MAX;
				if (do_mutation_gpu[i] < MUTATION_PROBABILITY)
				{
					mutation_step_x_gpu[i] = ((float)rand()) / (float)RAND_MAX;
					mutation_step_y_gpu[i] = ((float)rand()) / (float)RAND_MAX;
					mutation_max_step_host[i] = Mutation_step_max * pow(Mutation_annealing, gen);
				}
			}

			//Copies of random vectors on device 
			thrust::device_vector<float> device_do_crossover_gpu(do_crossover_gpu);
			thrust::device_vector<float> device_cross_x_gpu(cross_x_gpu);
			thrust::device_vector<float> device_cross_y_gpu(cross_y_gpu);
			thrust::device_vector<float> parent1_rand_gpu(parent1_rand);
			thrust::device_vector<float> parent2_rand_gpu(parent2_rand);

			thrust::device_vector<float> device_do_mutation_gpu(do_mutation_gpu);
			thrust::device_vector<float> device_mutation_step_x_gpu(mutation_step_x_gpu);
			thrust::device_vector<float> device_mutation_step_y_gpu(mutation_step_y_gpu);
			thrust::device_vector<float> device_mutation_step_maximum(mutation_max_step_host);

			//Pointers on copies of random vectors on device 
			float* device_do_crossover_gpu_pointer = thrust::raw_pointer_cast(&device_do_crossover_gpu[0]);
			float* device_cross_x_gpu_pointer = thrust::raw_pointer_cast(&device_cross_x_gpu[0]);
			float* device_cross_y_gpu_pointer = thrust::raw_pointer_cast(&device_cross_y_gpu[0]);
			float* parent1_rand_gpu_pointer = thrust::raw_pointer_cast(&parent1_rand_gpu[0]);
			float* parent2_rand_gpu_pointer = thrust::raw_pointer_cast(&parent2_rand_gpu[0]);

			float* device_do_mutation_gpu_pointer = thrust::raw_pointer_cast(&device_do_mutation_gpu[0]);
			float* device_mutation_step_x_gpu_pointer = thrust::raw_pointer_cast(&device_mutation_step_x_gpu[0]);
			float* device_mutation_step_y_gpu_pointer = thrust::raw_pointer_cast(&device_mutation_step_y_gpu[0]);
			float* device_mutation_step_maximum_pointer = thrust::raw_pointer_cast(&device_mutation_step_maximum[0]);

			//Crossover operation
			Crossover_gpu << < How_many_blocks, THREADS_PER_BLOCK >> > (thrust_pointer, thrust_pointer_new, device_do_crossover_gpu_pointer, device_cross_x_gpu_pointer, device_cross_y_gpu_pointer, parent1_rand_gpu_pointer, parent2_rand_gpu_pointer);

			//Mutation operation
			Mutation_gpu << < How_many_blocks, THREADS_PER_BLOCK >> > (thrust_pointer_new, device_do_mutation_gpu_pointer, device_mutation_step_x_gpu_pointer, device_mutation_step_y_gpu_pointer, device_mutation_step_maximum_pointer);

			host_thrust_member = device_thrust_member_new;

			if (Do_show_members_gpu == 1)
			{
				cout << "New memebers" << endl;
				Member* Pointer_to_show_members = thrust::raw_pointer_cast(&host_thrust_member[0]);
				Show_Population(Pointer_to_show_members);
			}
			if (Do_save_to_file == 1)
			{
				for (int i = 0; i < How_many_members-1; i++)
				{
					file_gpu << host_thrust_member[i].x << "," << host_thrust_member[i].y << ",";
				}
				file_gpu << host_thrust_member[How_many_members-1].x << "," << host_thrust_member[How_many_members-1].y << "\n";
			}

		}

		//Finding optimum

		device_thrust_member = host_thrust_member;
		Count_Fitness_gpu << < How_many_blocks, THREADS_PER_BLOCK >> > (thrust_pointer);
		thrust::sort(device_thrust_member.begin(), device_thrust_member.end());	//Sorting
		host_thrust_member = device_thrust_member;

		float x_opt_gpu  = 0;
		float y_opt_gpu = 0;

		for (int i = 0; i < how_many_included_average; i++)
		{
			x_opt_gpu += host_thrust_member[i].x;
			y_opt_gpu += host_thrust_member[i].y;
		}
		x_opt_gpu /= how_many_included_average;
		y_opt_gpu /= how_many_included_average;

		cout << "Optimum in x = " << x_opt_gpu << " y = " << y_opt_gpu << endl;

		Member Member_opt_gpu;
		Member_opt_gpu.x = x_opt_gpu;
		Member_opt_gpu.y = y_opt_gpu;
		Member_opt_gpu.fitness = FitFunction(Member_opt_gpu);

		cout << "Value of funtion in optimum: " << -Member_opt_gpu.fitness << endl;


		if (Do_save_to_file == 1)
		{
			file_gpu << x_opt_gpu << "," << y_opt_gpu << "\n";
			file_gpu << range_min_x << "," << range_max_x << "\n" << range_min_y << "," << range_max_y << "\n";


			if (file_gpu.good() == true)
			{
				cout << "Data successfully saved to file" << endl;
			}
			else
			{
				cout << "Access to the file was forbidden!" << std::endl;
				PauseSystem();
			}

			file_gpu.close();
		}

		cudaFree(Population_gpu_dev);
		delete[] Population_gpu;

		high_resolution_clock::time_point end_time_gpu = high_resolution_clock::now();
		duration_gpu = duration_cast<duration<double>>(end_time_gpu - begin_time_gpu);

		cout << "Processing time on GPU " << duration_gpu.count() << endl << endl;

	}

	if(do_run_on_cpu==1 && do_run_on_gpu==1)
		cout << "Speedup: " << duration_cpu.count() / duration_gpu.count() << endl;

	PauseSystem();
	return 0;

}
