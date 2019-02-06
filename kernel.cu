//Bartosz Bielinski - Genetic Algorithm

/*
	Description of algorithm:
	- First population is generated randomly
	- Fitness is estimated based on value of selected optimized function in point (x,y)
	- Members are sorted on basis of value of fitness and then each member is getting probability based on "rank based selection", which can be adjusted with parameters 'RANK_STEP_DOWN' and 'CROSSOVER_PROBABILITY'
	- Roulette selection algorithm is used to choose random members
	- To crossover process the "Blend Crossover" for Real-Coded Genetic Algorithms is used and it can be adjusted with parameter 'alpha' 
	- In case of mutation, the mutated member is moved along x and y axis by radom value from range <-Mutation_step_max, Mutation_step_max>, parameter to adjust: 'MUTATION_PROBABILITY'
	- Optimum is calculated as an average from a few best members, the amount of members included in average can be adjusted with parameter 'how_many_included_average' (default: 1) 
	- Each generation is saved to csv file, for CPU to file 'osobniki.csv', and for GPU to 'osobniki_gpu.csv'.
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
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cmath>
#include <cstdio>
#include <stdlib.h>    
#include <vector>
#include <time.h>       /* time */
#include <algorithm>    // std::sort
#include <fstream>
#include <chrono>
#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust\device_vector.h>
#include <thrust\sort.h>

using namespace std;
using namespace std::chrono;

//Parametry do dostosowania
#define range_min_x (float) -10
#define range_max_x (float) 10
#define range_min_y (float) -10
#define range_max_y (float) 10
#define WHICH_FUNCTION 3
#define Do_show_members_cpu 0
#define Do_show_members_gpu 0
#define Do_show_generations 0
#define Do_save_to_file 1
#define do_run_on_cpu 1
#define do_run_on_gpu 1
#define how_many_included_average 1

#define How_many_members 100
#define How_many_generations 30
//#define Mutation_step_min (float)0.1
#define Mutation_step_max (float)20
#define alpha (float)0.1
#define RANK_STEP_DOWN (float)0.3
#define CROSSOVER_PROBABILITY (float)0.7
#define MUTATION_PROBABILITY (float)0.1

#define BLOCKS_PER_KERNEL 1000
#define THREADS_PER_BLOCK 1024

struct Member {

	float x=0;
	float y=0;
	float fitness=0;
	float probability=0;

};


float FitFunction(Member member);
bool compareByFit(const Member &a, const Member &b);
bool compareByFit_gpu(const Member &a, const Member &b);
void Show_Population(vector <Member> Population);
void Show_Population(Member *Population);
float RandomFloat(float a, float b);
int Roulete_Selection(vector <Member> Population);
void Mutate(Member &member);
void Count_Fitness(vector <Member> &Population);
void Generate_Population(vector <Member> &Population);


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
			//Losowanie rodzica1
			float offset = 0.0;
			//Losowanie ruletkowe
			for (int i = 0; i < How_many_members; i++)
			{
				offset += Population[i].probability;
				if (offset > parent1_rand[index])
				{
					parent1 = i;
					break;
				}
			}

			//Losowanie rodzica2
			offset = 0.0;
			int parent2 = 0;

			//Losowanie ruletkowe
			for (int i = 0; i < How_many_members; i++)
			{
				offset += Population[i].probability;
				if (offset > parent2_rand[index])
				{
					parent2 = i;
					break;
				}
			}

			//Posortowanie rodzicow rosnaca - rodzic1 to ten wiekszy, a rodzic2 mniejszy
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

__global__ void Mutation_gpu(Member * thrust_pointer_new, float * do_mutation, float * rand_cross_x, float * rand_cross_y)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < How_many_members)
	{
		if (do_mutation[index] < MUTATION_PROBABILITY)
		{
			thrust_pointer_new[index].x = thrust_pointer_new[index].x + (-Mutation_step_max + (rand_cross_x[index])*(Mutation_step_max - (-Mutation_step_max)));
			
			if (thrust_pointer_new[index].x > range_max_x)
				thrust_pointer_new[index].x = range_max_x;
			if (thrust_pointer_new[index].x < range_min_x)
				thrust_pointer_new[index].x= range_min_x;

			
			thrust_pointer_new[index].y = thrust_pointer_new[index].y + (-Mutation_step_max + (rand_cross_y[index])*(Mutation_step_max - (-Mutation_step_max)));

			if (thrust_pointer_new[index].y > range_max_y)
				thrust_pointer_new[index].y = range_max_y;
			if (thrust_pointer_new[index].y < range_min_y)
				thrust_pointer_new[index].y = range_min_y;


		}
	}


}



int main()
{
	srand(time(NULL));

	cout << "Bartosz Bielinski - Algorytm Genetyczny" << endl;

	duration<double> duration_gpu;
	duration<double> duration_cpu;

	if (do_run_on_cpu == 1)
	{
		ofstream file;

		cout << "Liczenie na CPU" << endl;

		if (Do_save_to_file == 1)
		{
			file.open("osobniki.csv", std::ios::trunc);
			if (file.good() == true)
			{
				cout << "Uzyskano dostep do pliku!" << endl;
				file << How_many_members << "," << WHICH_FUNCTION << "\n";

			}
			else
			{
				cout << "Dostep do pliku zostal zabroniony!" << std::endl;
				system("PAUSE");
			}
		}

		//==============================================================================================================================
		//
		//
		// Algorytm Genetyczny na CPU
		//
		//
		//==============================================================================================================================


		high_resolution_clock::time_point begin_time_cpu = high_resolution_clock::now();

		vector <Member> Population;
		vector <Member> Population_new;

		//Generowanie populacji
		Generate_Population(Population);

		cout << "Rozmiar populacji wynosi " << Population.size() << endl;
		cout << "Ilosc generacji wynosi: " << How_many_generations << endl;

		if (Do_show_members_cpu == 1)
			Show_Population(Population);

		//Przebieg wielu pokolen populacji
		for (int gen = 0; gen < How_many_generations - 1; gen++)
		{
			if (Do_show_generations == 1)
				cout << endl << "Generacja: " << gen + 1 << endl << endl;;

			Count_Fitness(Population);

			if (Do_show_members_cpu == 1)
				Show_Population(Population);

			if (Do_save_to_file == 1)
			{
				for (int j = 0; j < Population.size(); j++)
				{
					file << Population[j].x << "," << Population[j].y << "\n";
				}
			}


			//Sortowanie osobnikow malejaco
			sort(Population.begin(), Population.end(), compareByFit);

			if (Do_show_members_cpu == 1)
			{
				cout << endl << "Posortowana populacja:" << endl;
				Show_Population(Population);
			}

			//Przypisywanie osobnikom prawdopodobnienstwa na podstawie rangi wybrania na podstawie rankingu - kolejnosci
			for (int j = 0; j < Population.size(); j++)
			{

				if (j != (Population.size() - 1))
					Population[j].probability = RANK_STEP_DOWN * pow((1 - RANK_STEP_DOWN), j);
				else if (j == (Population.size() - 1))
					Population[j].probability = pow((1 - RANK_STEP_DOWN), j);
			}

			if (Do_show_members_cpu == 1)
				Show_Population(Population);

			//Crossover dla każdego osobnika
			for (int i = 0; i < Population.size(); i++)
			{
				float do_crossover = ((float)rand()) / (float)RAND_MAX;
				float do_mutation = ((float)rand()) / (float)RAND_MAX;

				if (do_crossover < CROSSOVER_PROBABILITY)
				{
					int parent1 = Roulete_Selection(Population);
					int parent2 = Roulete_Selection(Population);

					//Posortowanie rodzicow rosnaca - rodzic1 to ten wiekszy, a rodzic2 mniejszy
					if (Population[parent1].fitness < Population[parent2].fitness)
					{
						int temp = parent1;
						parent1 = parent2;
						parent2 = temp;

					}

					float x_min = Population[parent2].x - alpha * (Population[parent1].x - Population[parent2].x);
					float x_max = Population[parent2].x + alpha * (Population[parent1].x - Population[parent2].x);
					float cross_x = RandomFloat(x_min, x_max);

					if (cross_x > range_max_x)
						cross_x = range_max_x;
					if (cross_x < range_min_x)
						cross_x = range_min_x;

					float y_min = Population[parent2].y - alpha * (Population[parent1].y - Population[parent2].y);
					float y_max = Population[parent2].y + alpha * (Population[parent1].y - Population[parent2].y);
					float cross_y = RandomFloat(y_min, y_max);

					if (cross_y > range_max_y)
						cross_y = range_max_y;
					if (cross_y < range_min_y)
						cross_y = range_min_y;

					Member child;
					child.x = cross_x;
					child.y = cross_y;
					//dziecko.fitness = 0;
					//dziecko.probability = 0;

					//Mutacja 
					if (do_mutation < MUTATION_PROBABILITY)
						Mutate(child);

					Population_new.push_back(child);

				}
				else
				{
					int which_member_unchanged = Roulete_Selection(Population);
					Member unchanged_member = Population[which_member_unchanged];

					if (do_mutation < MUTATION_PROBABILITY)
						Mutate(unchanged_member);

					Population_new.push_back(unchanged_member);

				}
			}

			//Zapisanie nowej generacji 
			Population = Population_new;
			Population_new.clear();

		}

		//Przebieg ostatniego pokolenia
		if (Do_show_generations == 1)
			cout << endl << "Generacja: " << How_many_generations << endl << endl;;

		Count_Fitness(Population);

		if (Do_show_members_cpu == 1)
			Show_Population(Population);

		if (Do_save_to_file == 1)
		{
			for (int j = 0; j < Population.size(); j++)
			{
				file << Population[j].x << "," << Population[j].y << "\n";
			}
		}


		//Znajdywanie optimum z ilosci osobnikow uzaleznionych od zmiennej ile_wliczonych
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

		cout << "Optimum w x = " << x_opt << " y = " << y_opt << endl;

		Member member_opt;
		member_opt.x = x_opt;
		member_opt.y = y_opt;
		member_opt.fitness = FitFunction(member_opt);

		cout << "Wartosc funkcji w optimum: " << -member_opt.fitness << endl;

		if (Do_save_to_file == 1)
		{
			file << x_opt << "," << y_opt << "\n";
			file << range_min_x << "," << range_max_x << "\n" << range_min_y << "," << range_max_y << "\n";

			if (file.good() == true)
			{
				cout << "Dane pomyslnie zapisane do pliku" << endl;
			}
			else
			{
				cout << "Dostep do pliku zostal zabroniony!" << std::endl;
				system("PAUSE");
			}

			file.close();
		}

		high_resolution_clock::time_point end_time_cpu = high_resolution_clock::now();

		duration_cpu = duration_cast<duration<double>>(end_time_cpu - begin_time_cpu);

		cout << "Czas liczenia na CPU: " << duration_cpu.count() << endl << endl;

		system("pause");

	}

	//==============================================================================================================================
	//
	//
	// Algorytm Genetyczny na GPU
	//
	//
	//==============================================================================================================================

	if (do_run_on_gpu == 1)
	{
		cout << "Liczenie na GPU" << endl;

		ofstream file_gpu;

		if (Do_save_to_file == 1)
		{
			file_gpu.open("osobniki_gpu.csv", std::ios::trunc);
			if (file_gpu.good() == true)
			{
				cout << "Uzyskano dostep do pliku!" << endl;
				file_gpu << How_many_members << "," << WHICH_FUNCTION << "\n";

			}
			else
			{
				cout << "Dostep do pliku zostal zabroniony!" << std::endl;
				system("PAUSE");
			}
		}

		high_resolution_clock::time_point begin_time_gpu = high_resolution_clock::now();

		Member *Population_gpu = new Member[How_many_members];
		Member *Population_gpu_dev;

		unsigned long long int How_many_blocks = How_many_members / THREADS_PER_BLOCK + 1;
		cout << "Potrzebnych blokow na GPU: " << How_many_blocks << endl << endl;

		//Wypisz populacje
		if (Do_show_members_gpu == 1)
			Show_Population(Population_gpu);

		int size = How_many_members * sizeof(Member);
		cudaMalloc(&Population_gpu_dev, size);
		cudaMemcpy(Population_gpu_dev, Population_gpu, size, cudaMemcpyHostToDevice);

		//Generowanie losowych liczb potrzebnych do wygenerowania populacji
		float *Random_member_x = new float[How_many_members];
		for (int i = 0; i < How_many_members; i++)
			Random_member_x[i] = RandomFloat(range_min_x, range_max_x);

		float *dev_Random_member_x;
		cudaMalloc(&dev_Random_member_x, How_many_members * sizeof(float));
		cudaMemcpy(dev_Random_member_x, Random_member_x, How_many_members * sizeof(float), cudaMemcpyHostToDevice);

		//Generowanie losowych liczb potrzebnych do wygenerowania populacji
		float *Random_member_y = new float[How_many_members];
		for (int i = 0; i < How_many_members; i++)
			Random_member_y[i] = RandomFloat(range_min_x, range_max_x);

		float *dev_Random_member_y;
		cudaMalloc(&dev_Random_member_y, How_many_members * sizeof(float));
		cudaMemcpy(dev_Random_member_y, Random_member_y, How_many_members * sizeof(float), cudaMemcpyHostToDevice);

		//Generowanie populacji
		Generate_Population_gpu << <How_many_blocks, THREADS_PER_BLOCK >> > (Population_gpu_dev, dev_Random_member_x, dev_Random_member_y);

		cudaMemcpy(Population_gpu, Population_gpu_dev, size, cudaMemcpyDeviceToHost);

		cudaFree(dev_Random_member_x);
		cudaFree(dev_Random_member_y);

		delete[]Random_member_x;
		delete[]Random_member_y;

		if (Do_show_generations == 1)
			cout << endl << "Generacja: 1" << endl << endl;

		if (Do_show_members_gpu == 1)
			Show_Population(Population_gpu);

		//Tworzenie wektorow thrust do przechowywania populacji na host i device
		thrust::host_vector<Member> host_thrust_member(How_many_members);
		thrust::device_vector<Member> device_thrust_member(host_thrust_member);
		Member* thrust_pointer = thrust::raw_pointer_cast(&device_thrust_member[0]);	//Wskaznik na miejsce w pamieci na device gdzie przechowywana jest populacja, czyli na poczatek wektoru


		for (int i = 0; i < How_many_members; i++)
		{
			host_thrust_member[i] = Population_gpu[i];
		}

		if (Do_save_to_file == 1)
		{
			for (int i = 0; i < How_many_members; i++)
			{
				file_gpu << host_thrust_member[i].x << "," << host_thrust_member[i].y << "\n";
			}
		}

		//Wektory w ktorych beda przechowywane losowe liczby potrzebne do operacji crossover
		thrust::host_vector<float> do_crossover_gpu(How_many_members);
		thrust::host_vector<float> cross_x_gpu(How_many_members);
		thrust::host_vector<float> cross_y_gpu(How_many_members);
		thrust::host_vector<float> parent1_rand(How_many_members);
		thrust::host_vector<float> parent2_rand(How_many_members);


		//Wektory w ktorych beda przechowywane losowe liczby potrzebne do operacji mutacji
		thrust::host_vector<float> do_mutation_gpu(How_many_members);
		thrust::host_vector<float> mutation_step_x_gpu(How_many_members);
		thrust::host_vector<float> mutation_step_y_gpu(How_many_members);


		//Przebieg wielu pokolen populacji
		for (int gen = 0; gen < How_many_generations - 1; gen++)
		{
			if (Do_show_generations == 1)
				cout << endl << "Generacja: " << gen + +2 << endl << endl;

			//Kopiowanie osobnikow z hosta na device
			device_thrust_member = host_thrust_member;

			if (Do_show_members_gpu == 1)
			{
				cout << "Nie policzone" << endl;
				Member* Pointer_to_show_members = thrust::raw_pointer_cast(&host_thrust_member[0]);
				Show_Population(Pointer_to_show_members);
			}

			//Liczenie fitness dla kazdego osobnika w populacji
			Count_Fitness_gpu << < How_many_blocks, THREADS_PER_BLOCK >> > (thrust_pointer);

			if (Do_show_members_gpu == 1)
			{
				host_thrust_member = device_thrust_member;		//Kopiowanie populacji z device na hosta

				cout << "Policzone" << endl;
				Member* Pointer_to_show_members = thrust::raw_pointer_cast(&host_thrust_member[0]);
				Show_Population(Pointer_to_show_members);
			}


			//Sortowanie osobnikow w populacji malejaco wedlug fitnessu
			thrust::sort(device_thrust_member.begin(), device_thrust_member.end());

			if (Do_show_members_gpu == 1)
			{
				host_thrust_member = device_thrust_member;

				cout << endl << "Posortowane: " << endl;
				Member* Pointer_to_show_members = thrust::raw_pointer_cast(&host_thrust_member[0]);
				Show_Population(Pointer_to_show_members);
			}


			//Przypisywanie osobnikom prawdopodobnienstwa na podstawie rangi wybrania na podstawie rankingu - kolejnosci
			Count_Probability << < How_many_blocks, THREADS_PER_BLOCK >> > (thrust_pointer);

			host_thrust_member = device_thrust_member;

			if (Do_show_members_gpu == 1)
			{
				cout << "z Policzonym prawdp" << endl;
				Member* Pointer_to_show_members = thrust::raw_pointer_cast(&host_thrust_member[0]);
				Show_Population(Pointer_to_show_members);
			}


			//Crossover

			//Wektory na device w ktorych bedzie przechowywana nowa populacja juz po crossover
			thrust::device_vector<Member> device_thrust_member_new(host_thrust_member);
			Member* thrust_pointer_new = thrust::raw_pointer_cast(&device_thrust_member_new[0]);

			//Generowanie liczb losowych potrzebnych w kernelach
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
				}
			}

			//Kopie wektorów losowych na device
			thrust::device_vector<float> device_do_crossover_gpu(do_crossover_gpu);
			thrust::device_vector<float> device_cross_x_gpu(cross_x_gpu);
			thrust::device_vector<float> device_cross_y_gpu(cross_y_gpu);
			thrust::device_vector<float> parent1_rand_gpu(parent1_rand);
			thrust::device_vector<float> parent2_rand_gpu(parent2_rand);

			thrust::device_vector<float> device_do_mutation_gpu(do_mutation_gpu);
			thrust::device_vector<float> device_mutation_step_x_gpu(mutation_step_x_gpu);
			thrust::device_vector<float> device_mutation_step_y_gpu(mutation_step_y_gpu);

			//Wskaźniki na kopie wektorów losowych na device
			float* device_do_crossover_gpu_pointer = thrust::raw_pointer_cast(&device_do_crossover_gpu[0]);
			float* device_cross_x_gpu_pointer = thrust::raw_pointer_cast(&device_cross_x_gpu[0]);
			float* device_cross_y_gpu_pointer = thrust::raw_pointer_cast(&device_cross_y_gpu[0]);
			float* parent1_rand_gpu_pointer = thrust::raw_pointer_cast(&parent1_rand_gpu[0]);
			float* parent2_rand_gpu_pointer = thrust::raw_pointer_cast(&parent2_rand_gpu[0]);

			float* device_do_mutation_gpu_pointer = thrust::raw_pointer_cast(&device_do_mutation_gpu[0]);
			float* device_mutation_step_x_gpu_pointer = thrust::raw_pointer_cast(&device_mutation_step_x_gpu[0]);
			float* device_mutation_step_y_gpu_pointer = thrust::raw_pointer_cast(&device_mutation_step_y_gpu[0]);

			//Operacja crossover
			Crossover_gpu << < How_many_blocks, THREADS_PER_BLOCK >> > (thrust_pointer, thrust_pointer_new, device_do_crossover_gpu_pointer, device_cross_x_gpu_pointer, device_cross_y_gpu_pointer, parent1_rand_gpu_pointer, parent2_rand_gpu_pointer);

			//Operacja mutacji
			Mutation_gpu << < How_many_blocks, THREADS_PER_BLOCK >> > (thrust_pointer_new, device_do_mutation_gpu_pointer, device_mutation_step_x_gpu_pointer, device_mutation_step_y_gpu_pointer);

			host_thrust_member = device_thrust_member_new;

			if (Do_show_members_gpu == 1)
			{
				cout << "Nowe osobniki" << endl;
				Member* Pointer_to_show_members = thrust::raw_pointer_cast(&host_thrust_member[0]);
				Show_Population(Pointer_to_show_members);
			}
			if (Do_save_to_file == 1)
			{
				for (int i = 0; i < How_many_members; i++)
				{
					file_gpu << host_thrust_member[i].x << "," << host_thrust_member[i].y << "\n";
				}
			}

		}

		//Znajdywanie optimum

		device_thrust_member = host_thrust_member;
		Count_Fitness_gpu << < How_many_blocks, THREADS_PER_BLOCK >> > (thrust_pointer);
		thrust::sort(device_thrust_member.begin(), device_thrust_member.end());	//Sortowanie
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

		cout << "Optimum w x = " << x_opt_gpu << " y = " << y_opt_gpu << endl;

		Member Member_opt_gpu;
		Member_opt_gpu.x = x_opt_gpu;
		Member_opt_gpu.y = y_opt_gpu;
		Member_opt_gpu.fitness = FitFunction(Member_opt_gpu);

		cout << "Wartosc funkcji w optimum: " << -Member_opt_gpu.fitness << endl;


		if (Do_save_to_file == 1)
		{
			file_gpu << x_opt_gpu << "," << y_opt_gpu << "\n";
			file_gpu << range_min_x << "," << range_max_x << "\n" << range_min_y << "," << range_max_y << "\n";


			if (file_gpu.good() == true)
			{
				cout << "Dane pomyslnie zapisane do pliku" << endl;
			}
			else
			{
				cout << "Dostep do pliku zostal zabroniony!" << std::endl;
				system("PAUSE");
			}

			file_gpu.close();
		}

		cudaFree(Population_gpu_dev);
		delete[] Population_gpu;

		high_resolution_clock::time_point end_time_gpu = high_resolution_clock::now();
		duration_gpu = duration_cast<duration<double>>(end_time_gpu - begin_time_gpu);

		cout << "Czas liczenia na GPU " << duration_gpu.count() << endl << endl;

	}

	if(do_run_on_cpu==1 && do_run_on_gpu==1)
		cout << "Speedup: " << duration_cpu.count() / duration_gpu.count() << endl;

	system("pause");
	return 0;

}

	//==============================================================================================================================
	//
	//
	// Funkcje zewnętrzne
	//
	//
	//==============================================================================================================================


void Generate_Population(vector <Member> &Population)
{
	for (int i = 0; i < How_many_members; i++)
	{
		float x_rand = RandomFloat(range_min_x, range_max_x);
		float y_rand = RandomFloat(range_min_y, range_max_y);

		Member member;
		member.x = x_rand;
		member.y = y_rand;
		//osobnik.fitness = 0;
		//osobnik.probability = 0;

		Population.push_back(member);
	}

	return;
}


float FitFunction(Member member)
{
	float fittness = 0;
	if (WHICH_FUNCTION == 1)
		fittness = -(sin(member.x) + cos(member.y));
	else if (WHICH_FUNCTION == 2)
		fittness = -((member.x)*(member.x) + (member.y)*(member.y));
	else if (WHICH_FUNCTION == 3)
		fittness = -((member.y)*sin(member.x) - (member.x)*cos(member.y));
	else if (WHICH_FUNCTION == 4)
		fittness = -(sin(member.x + member.y) + (member.x - member.y)*(member.x - member.y) - 1.5*member.x + 2.5*member.y + 1);
	else if (WHICH_FUNCTION == 5)
		fittness = -(-(member.y + 47)*sin(sqrt(abs((member.x / 2) + (member.y + 47)))) - member.x*sin(sqrt(abs(member.x - (member.y + 47)))));
	else if (WHICH_FUNCTION == 6)
		fittness = -(418.9829 * 2 - (member.x*sin(sqrt(abs(member.x))) + member.y*sin(sqrt(abs(member.y)))));
	return fittness;
}


bool compareByFit(const Member &a, const Member &b)
{
	return a.fitness > b.fitness;
}

bool compareByFit_gpu(const Member &a, const Member &b)
{
	return a.fitness > b.fitness;
}

void Show_Population(vector <Member> Population)
{
	cout << endl;
	for (int j = 0; j < Population.size(); j++)
	{
		cout << "Osobnik " << j + 1 << ": x=" << Population[j].x << "  y=" << Population[j].y << " fit= " << Population[j].fitness <<" prob: "<<Population[j].probability<< endl;
	}
	cout << endl;
}

void Show_Population(Member *Population)
{
	cout << endl;
	for (int j = 0; j < How_many_members; j++)
	{
		cout << "Osobnik " << j + 1 << ": x=" << Population[j].x << "  y=" << Population[j].y << " fit= " << Population[j].fitness << " prob: " << Population[j].probability << endl;
	}
	cout << endl;
}

float RandomFloat(float a, float b)
{
	float random = ((float)rand()) / (float)RAND_MAX;
	float diff = b - a;
	float r = random * diff;
	return a + r;
}

int Roulete_Selection(vector <Member> Population)
{
	float offset = 0.0;
	float rand_cross = ((float)rand()) / (float)RAND_MAX;
	int parent = 0;

	//Losowanie ruletkowe
	for (int j = 0; j < Population.size(); j++)
	{
		offset += Population[j].probability;
		if (offset > rand_cross)
		{
			parent = j;
			break;
		}
	}

	return parent;
}

void Mutate(Member &member)
{

	float Mutation_step_x = RandomFloat(-Mutation_step_max, Mutation_step_max);
	float Mutation_step_y = RandomFloat(-Mutation_step_max, Mutation_step_max);

	member.x = member.x + Mutation_step_x;
	if (member.x > range_max_x)
		member.x = range_max_x;
	if (member.x < range_min_x)
		member.x = range_min_x;

	member.y = member.y + Mutation_step_y;
	if (member.y > range_max_y)
		member.y = range_max_y;
	if (member.y < range_min_y)
		member.y = range_min_y;


	return;
}

void Count_Fitness(vector <Member> &Population)
{
	//Obliczanie fitness dla każdego osobnika
	for (int j = 0; j < Population.size(); j++)
	{
		Population[j].fitness = FitFunction(Population[j]);
	}

	return;
}

