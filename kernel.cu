//Bartosz Bielinski - Algorytm Genetyczny

/*
	Opis algorytmu:
	- Pierwsza populacja jest generowana całkowicie losowo
	- Fitness jest liczony na podstawie wartości wybranej optymalizowanej funkcji w danym punkcie (x,y)
	- Osobniki są oceniane na podstawie wartości fitnessu i jest im przypisywane prawdopodobieństwo na podstawie metody "rank based selection", która może być optymalizowany za pomocą parametru 'RANK_STEP_DOWN' i 'CROSSOVER_PROBABILITY'
	- Do losowania osobników używany jest algorytm losowania ruletkowego
	- Do procesu crossover wykorzystywana jest metoda "Blend Crossover" dla Real-Coded Genetic Algorithms i może być on optymalizowany za pomocą parametru 'alpha'
	- W przypadku mutacji, mutowany osobnik jest przesuwany w osiach x i y o losową wartość z przedziału <-Mutation_step_max, Mutation_step_max>, parametry do optymalizowania: 'MUTATION_PROBABILITY'
	- Optimum jest wyliczane jako srednia z kilku najlepszych osobnikow na końcu, ilość branych pod uwagę osobników jest zmieniana za pomocą zmiennej 'ile_wliczonych', domyślnie 1.
	- Dane są zapisane do plików zewnętrznych w formacie .csv, dla CPU do pliku 'osobniki.csv', a dla GPU do 'osobniki_gpu.csv'.
*/
/*
	Funkcje do wyboru:
	1. f(x,y) = sin(x) + cos(x) - brak optimum globalnego - dowolny przedział
	2. f(x,y) = x^2 + y^2 - parabola w 3D (taka mistka), minium globalne w (0,0), f(0,0) = 0 - dowolny przedział
	3. f(x,y) = y*sin(x) - x*sin(y) - optimum do znalezienia wizualnie, dowolny przedział
	4. McCormick Function - f(x,y) =  sin(x+y) + (x-y)^2 -1.5x +2.5y +1, minimum globalne na przedziale -1.5<=x<=4  -3<=y<=4  w punkcie (-0.54719, -1.54719) i f = -1.9133
	5. Eggholder function - na przedziale  -512<=x,y<=512 jest minimum globalne w (512, 404.2319) i f = -959.6407
	6. Schwefel Function - na przedziale -500<=x,y<=500 ma minimum globalne w (420.9687, 420.9687) i f = 0.
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
#define zakres_min_x (float) -500
#define zakres_max_x (float) 500
#define zakres_min_y (float) -500
#define zakres_max_y (float) 500
#define KTORA_FUNKCJA 6
#define Czy_wyswietl_osobniki_cpu 0
#define Czy_wyswietl_osobniki_gpu 0
#define Czy_wyswietl_generacje 0
#define czy_zapisac_do_pliku 1
#define czy_licz_cpu 1
#define czy_licz_gpu 1
#define ile_wliczonych 1

#define ILE_OSOBNIKOW 5000
#define ILE_GENERACJI 100
//#define Mutation_step_min (float)0.1
#define Mutation_step_max (float)1000
#define alpha (float)0.1
#define RANK_STEP_DOWN (float)0.3
#define CROSSOVER_PROBABILITY (float)0.7
#define MUTATION_PROBABILITY (float)0.1

#define BLOCKS_PER_KERNEL 1000
#define THREADS_PER_BLOCK 1024

struct Osobnik {

	float x=0;
	float y=0;
	float fitness=0;
	float probability=0;

};


float FitFunction(Osobnik osobnik);
bool compareByFit(const Osobnik &a, const Osobnik &b);
bool compareByFit_gpu(const Osobnik &a, const Osobnik &b);
void Wypisz_Populacje(vector <Osobnik> Populacja);
void Wypisz_Populacje(Osobnik *Populacja);
float RandomFloat(float a, float b);
int Roulete_Selection(vector <Osobnik> Populacja);
void Mutuj(Osobnik &osobnik);
void Policz_Fitness(vector <Osobnik> &Populacja);
void Generuj_Populacje(vector <Osobnik> &Populacja);


__host__ __device__ bool operator<(const Osobnik &lhs, const Osobnik &rhs)
{ return (lhs.fitness > rhs.fitness); };


__global__ void Generuj_Populacje_gpu(Osobnik *Populacja_gpu_dev, float *Random_x, float *Random_y)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < ILE_OSOBNIKOW)
	{
		Populacja_gpu_dev[index].x = Random_x[index];
		Populacja_gpu_dev[index].y = Random_y[index];
	}
}

__global__ void Policz_Fitness_gpu(Osobnik* Populacja_gpu_dev)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < ILE_OSOBNIKOW)
	{
		if (KTORA_FUNKCJA == 1)
			Populacja_gpu_dev[index].fitness = -(sin(Populacja_gpu_dev[index].x) + cos(Populacja_gpu_dev[index].y));
		else if (KTORA_FUNKCJA == 2)
			Populacja_gpu_dev[index].fitness = -((Populacja_gpu_dev[index].x)*(Populacja_gpu_dev[index].x) + (Populacja_gpu_dev[index].y)*(Populacja_gpu_dev[index].y));
		else if (KTORA_FUNKCJA == 3)
			Populacja_gpu_dev[index].fitness = -((Populacja_gpu_dev[index].y)*sin(Populacja_gpu_dev[index].x) - (Populacja_gpu_dev[index].x)*cos(Populacja_gpu_dev[index].y));
		else if (KTORA_FUNKCJA == 4)
			Populacja_gpu_dev[index].fitness = -(sin(Populacja_gpu_dev[index].x + Populacja_gpu_dev[index].y) + (Populacja_gpu_dev[index].x - Populacja_gpu_dev[index].y)*(Populacja_gpu_dev[index].x - Populacja_gpu_dev[index].y) - 1.5*Populacja_gpu_dev[index].x + 2.5*Populacja_gpu_dev[index].y + 1);
		else if (KTORA_FUNKCJA == 5)
			Populacja_gpu_dev[index].fitness = -(-(Populacja_gpu_dev[index].y + 47)*sin(sqrt(abs((Populacja_gpu_dev[index].x / 2) + (Populacja_gpu_dev[index].y + 47)))) - Populacja_gpu_dev[index].x*sin(sqrt(abs(Populacja_gpu_dev[index].x - (Populacja_gpu_dev[index].y + 47)))));
		else if (KTORA_FUNKCJA == 6)
			Populacja_gpu_dev[index].fitness = -(418.9829 * 2 - (Populacja_gpu_dev[index].x*sin(sqrt(abs(Populacja_gpu_dev[index].x))) + Populacja_gpu_dev[index].y*sin(sqrt(abs(Populacja_gpu_dev[index].y)))));

	}
}

__global__ void Policz_Prawdopodobienstwo(Osobnik* Populacja_gpu_dev)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < ILE_OSOBNIKOW)
	{
		if(index != (ILE_OSOBNIKOW-1))
			Populacja_gpu_dev[index].probability = RANK_STEP_DOWN * pow((1 - RANK_STEP_DOWN), index);
		else if (index == (ILE_OSOBNIKOW - 1))
			Populacja_gpu_dev[index].probability = pow((1 - RANK_STEP_DOWN), index);
	}
}

__global__ void Crossover_gpu(Osobnik *Populacja, Osobnik *Populacja_nowa, float *czy_crossover, float *rand_cross_x,float * rand_cross_y, float * rodzic1_rand, float * rodzic2_rand)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < ILE_OSOBNIKOW)
	{
		int rodzic1 = 0;

		if (czy_crossover[index] < CROSSOVER_PROBABILITY)
		{
			//Losowanie rodzica1
			float offset = 0.0;
			//Losowanie ruletkowe
			for (int i = 0; i < ILE_OSOBNIKOW; i++)
			{
				offset += Populacja[i].probability;
				if (offset > rodzic1_rand[index])
				{
					rodzic1 = i;
					break;
				}
			}

			//Losowanie rodzica2
			offset = 0.0;
			int rodzic2 = 0;

			//Losowanie ruletkowe
			for (int i = 0; i < ILE_OSOBNIKOW; i++)
			{
				offset += Populacja[i].probability;
				if (offset > rodzic2_rand[index])
				{
					rodzic2 = i;
					break;
				}
			}

			//Posortowanie rodzicow rosnaca - rodzic1 to ten wiekszy, a rodzic2 mniejszy
			if (Populacja[rodzic1].fitness < Populacja[rodzic2].fitness)
			{
				int temp = rodzic1;
				rodzic1 = rodzic2;
				rodzic2 = temp;
			}

			float x_min = Populacja[rodzic2].x - alpha * (Populacja[rodzic1].x - Populacja[rodzic2].x);
			float x_max = Populacja[rodzic2].x + alpha * (Populacja[rodzic1].x - Populacja[rodzic2].x);
			
			float cross_x = x_min + (rand_cross_x[index])*(x_max - x_min);

			if (cross_x > zakres_max_x)
				cross_x = zakres_max_x;
			if (cross_x < zakres_min_x)
				cross_x = zakres_min_x;

			float y_min = Populacja[rodzic2].y - alpha * (Populacja[rodzic1].y - Populacja[rodzic2].y);
			float y_max = Populacja[rodzic2].y + alpha * (Populacja[rodzic1].y - Populacja[rodzic2].y);
			
			float cross_y = y_min + (rand_cross_y[index])*(y_max - y_min);

			if (cross_y > zakres_max_y)
				cross_y = zakres_max_y;
			if (cross_y < zakres_min_y)
				cross_y = zakres_min_y;

			Populacja_nowa[index].x = cross_x;
			Populacja_nowa[index].y = cross_y;

		}
		else
		{
			Populacja_nowa[index] = Populacja[rodzic1];
		}
	}
}

__global__ void Mutacja_gpu(Osobnik * thrust_pointer_nowy, float * czy_mutacja, float * rand_cross_x, float * rand_cross_y)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < ILE_OSOBNIKOW)
	{
		if (czy_mutacja[index] < MUTATION_PROBABILITY)
		{
			thrust_pointer_nowy[index].x = thrust_pointer_nowy[index].x + (-Mutation_step_max + (rand_cross_x[index])*(Mutation_step_max - (-Mutation_step_max)));
			
			if (thrust_pointer_nowy[index].x > zakres_max_x)
				thrust_pointer_nowy[index].x = zakres_max_x;
			if (thrust_pointer_nowy[index].x < zakres_min_x)
				thrust_pointer_nowy[index].x= zakres_min_x;

			
			thrust_pointer_nowy[index].y = thrust_pointer_nowy[index].y + (-Mutation_step_max + (rand_cross_y[index])*(Mutation_step_max - (-Mutation_step_max)));

			if (thrust_pointer_nowy[index].y > zakres_max_y)
				thrust_pointer_nowy[index].y = zakres_max_y;
			if (thrust_pointer_nowy[index].y < zakres_min_y)
				thrust_pointer_nowy[index].y = zakres_min_y;


		}
	}


}



int main()
{
	srand(time(NULL));

	cout << "Bartosz Bielinski - Algorytm Genetyczny" << endl;

	duration<double> duration_gpu;
	duration<double> duration_cpu;

	if (czy_licz_cpu == 1)
	{
		ofstream plik;

		cout << "Liczenie na CPU" << endl;

		if (czy_zapisac_do_pliku == 1)
		{
			plik.open("osobniki.csv", std::ios::trunc);
			if (plik.good() == true)
			{
				cout << "Uzyskano dostep do pliku!" << endl;
				plik << ILE_OSOBNIKOW << "," << KTORA_FUNKCJA << "\n";

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

		vector <Osobnik> Populacja;
		vector <Osobnik> Populacja_nowa;

		//Generowanie populacji
		Generuj_Populacje(Populacja);

		cout << "Rozmiar populacji wynosi " << Populacja.size() << endl;
		cout << "Ilosc generacji wynosi: " << ILE_GENERACJI << endl;

		if (Czy_wyswietl_osobniki_cpu == 1)
			Wypisz_Populacje(Populacja);

		//Przebieg wielu pokolen populacji
		for (int gen = 0; gen < ILE_GENERACJI - 1; gen++)
		{
			if (Czy_wyswietl_generacje == 1)
				cout << endl << "Generacja: " << gen + 1 << endl << endl;;

			Policz_Fitness(Populacja);

			if (Czy_wyswietl_osobniki_cpu == 1)
				Wypisz_Populacje(Populacja);

			if (czy_zapisac_do_pliku == 1)
			{
				for (int j = 0; j < Populacja.size(); j++)
				{
					plik << Populacja[j].x << "," << Populacja[j].y << "\n";
				}
			}


			//Sortowanie osobnikow malejaco
			sort(Populacja.begin(), Populacja.end(), compareByFit);

			if (Czy_wyswietl_osobniki_cpu == 1)
			{
				cout << endl << "Posortowana populacja:" << endl;
				Wypisz_Populacje(Populacja);
			}

			//Przypisywanie osobnikom prawdopodobnienstwa na podstawie rangi wybrania na podstawie rankingu - kolejnosci
			for (int j = 0; j < Populacja.size(); j++)
			{

				if (j != (Populacja.size() - 1))
					Populacja[j].probability = RANK_STEP_DOWN * pow((1 - RANK_STEP_DOWN), j);
				else if (j == (Populacja.size() - 1))
					Populacja[j].probability = pow((1 - RANK_STEP_DOWN), j);
			}

			if (Czy_wyswietl_osobniki_cpu == 1)
				Wypisz_Populacje(Populacja);

			//Niepotrzebne bo prawdopodobnienstwa juz sie sumuja do 100%
			/*
			//Wyliczenie sumy wszystkich prawdopodobienstw z rangi
			float PROB_SUM = 0.0;
			for (int i = 0; i < Populacja.size(); i++)
			{
				PROB_SUM += Populacja[i].probability;
			}
			cout << endl << "suma prawd = " << PROB_SUM << endl;

			//Wyliczenie prawdopodobnienstw sumujacych się do 100%
			for (int i = 0; i < Populacja.size(); i++)
			{
				Populacja[i].probability = Populacja[i].probability / PROB_SUM;

				if (Czy_wyswietl_osobniki == 1)
					cout << "Osobnik " << i + 1 << ": x=" << Populacja[i].x << "  y=" << Populacja[i].y << " fit= " << Populacja[i].fitness << " prawd= " << Populacja[i].probability << endl;
			}
			*/

			//Crossover dla każdego osobnika
			for (int i = 0; i < Populacja.size(); i++)
			{
				float czy_crossover = ((float)rand()) / (float)RAND_MAX;
				float czy_mutuj = ((float)rand()) / (float)RAND_MAX;

				if (czy_crossover < CROSSOVER_PROBABILITY)
				{
					int rodzic1 = Roulete_Selection(Populacja);
					int rodzic2 = Roulete_Selection(Populacja);

					//Posortowanie rodzicow rosnaca - rodzic1 to ten wiekszy, a rodzic2 mniejszy
					if (Populacja[rodzic1].fitness < Populacja[rodzic2].fitness)
					{
						int temp = rodzic1;
						rodzic1 = rodzic2;
						rodzic2 = temp;

					}

					float x_min = Populacja[rodzic2].x - alpha * (Populacja[rodzic1].x - Populacja[rodzic2].x);
					float x_max = Populacja[rodzic2].x + alpha * (Populacja[rodzic1].x - Populacja[rodzic2].x);
					float cross_x = RandomFloat(x_min, x_max);

					if (cross_x > zakres_max_x)
						cross_x = zakres_max_x;
					if (cross_x < zakres_min_x)
						cross_x = zakres_min_x;

					float y_min = Populacja[rodzic2].y - alpha * (Populacja[rodzic1].y - Populacja[rodzic2].y);
					float y_max = Populacja[rodzic2].y + alpha * (Populacja[rodzic1].y - Populacja[rodzic2].y);
					float cross_y = RandomFloat(y_min, y_max);

					if (cross_y > zakres_max_y)
						cross_y = zakres_max_y;
					if (cross_y < zakres_min_y)
						cross_y = zakres_min_y;

					Osobnik dziecko;
					dziecko.x = cross_x;
					dziecko.y = cross_y;
					//dziecko.fitness = 0;
					//dziecko.probability = 0;

					//Mutacja 
					if (czy_mutuj < MUTATION_PROBABILITY)
						Mutuj(dziecko);

					Populacja_nowa.push_back(dziecko);

				}
				else
				{
					int ktory_osobnik_bez_zmian = Roulete_Selection(Populacja);
					Osobnik osobnik_bez_zmian = Populacja[ktory_osobnik_bez_zmian];

					if (czy_mutuj < MUTATION_PROBABILITY)
						Mutuj(osobnik_bez_zmian);

					Populacja_nowa.push_back(osobnik_bez_zmian);

				}
			}

			//Zapisanie nowej generacji 
			Populacja = Populacja_nowa;
			Populacja_nowa.clear();

		}

		//Przebieg ostatniego pokolenia
		if (Czy_wyswietl_generacje == 1)
			cout << endl << "Generacja: " << ILE_GENERACJI << endl << endl;;

		Policz_Fitness(Populacja);

		if (Czy_wyswietl_osobniki_cpu == 1)
			Wypisz_Populacje(Populacja);

		if (czy_zapisac_do_pliku == 1)
		{
			for (int j = 0; j < Populacja.size(); j++)
			{
				plik << Populacja[j].x << "," << Populacja[j].y << "\n";
			}
		}


		//Znajdywanie optimum z ilosci osobnikow uzaleznionych od zmiennej ile_wliczonych
		sort(Populacja.begin(), Populacja.end(), compareByFit);

		float x_opt = 0;
		float y_opt = 0;

		for (int i = 0; i < ile_wliczonych; i++)
		{
			x_opt += Populacja[i].x;
			y_opt += Populacja[i].y;
		}

		x_opt /= ile_wliczonych;
		y_opt /= ile_wliczonych;

		cout << "Optimum w x = " << x_opt << " y = " << y_opt << endl;

		Osobnik osobnik_opt;
		osobnik_opt.x = x_opt;
		osobnik_opt.y = y_opt;
		osobnik_opt.fitness = FitFunction(osobnik_opt);

		cout << "Wartosc funkcji w optimum: " << -osobnik_opt.fitness << endl;

		if (czy_zapisac_do_pliku == 1)
		{
			plik << x_opt << "," << y_opt << "\n";
			plik << zakres_min_x << "," << zakres_max_x << "\n" << zakres_min_y << "," << zakres_max_y << "\n";

			if (plik.good() == true)
			{
				cout << "Dane pomyslnie zapisane do pliku" << endl;
			}
			else
			{
				cout << "Dostep do pliku zostal zabroniony!" << std::endl;
				system("PAUSE");
			}

			plik.close();
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

	if (czy_licz_gpu == 1)
	{
		cout << "Liczenie na GPU" << endl;

		ofstream plik_gpu;

		if (czy_zapisac_do_pliku == 1)
		{
			plik_gpu.open("osobniki_gpu.csv", std::ios::trunc);
			if (plik_gpu.good() == true)
			{
				cout << "Uzyskano dostep do pliku!" << endl;
				plik_gpu << ILE_OSOBNIKOW << "," << KTORA_FUNKCJA << "\n";

			}
			else
			{
				cout << "Dostep do pliku zostal zabroniony!" << std::endl;
				system("PAUSE");
			}
		}

		high_resolution_clock::time_point begin_time_gpu = high_resolution_clock::now();

		Osobnik *Populacja_gpu = new Osobnik[ILE_OSOBNIKOW];
		Osobnik *Populacja_gpu_dev;

		unsigned long long int Ilosc_blokow = ILE_OSOBNIKOW / THREADS_PER_BLOCK + 1;
		cout << "Potrzebnych blokow na GPU: " << Ilosc_blokow << endl << endl;

		//Wypisz populacje
		if (Czy_wyswietl_osobniki_gpu == 1)
			Wypisz_Populacje(Populacja_gpu);

		int size = ILE_OSOBNIKOW * sizeof(Osobnik);
		cudaMalloc(&Populacja_gpu_dev, size);
		cudaMemcpy(Populacja_gpu_dev, Populacja_gpu, size, cudaMemcpyHostToDevice);

		//Generowanie losowych liczb potrzebnych do wygenerowania populacji
		float *Random_osobniki_x = new float[ILE_OSOBNIKOW];
		for (int i = 0; i < ILE_OSOBNIKOW; i++)
			Random_osobniki_x[i] = RandomFloat(zakres_min_x, zakres_max_x);

		float *dev_Random_osobniki_x;
		cudaMalloc(&dev_Random_osobniki_x, ILE_OSOBNIKOW * sizeof(float));
		cudaMemcpy(dev_Random_osobniki_x, Random_osobniki_x, ILE_OSOBNIKOW * sizeof(float), cudaMemcpyHostToDevice);

		//Generowanie losowych liczb potrzebnych do wygenerowania populacji
		float *Random_osobniki_y = new float[ILE_OSOBNIKOW];
		for (int i = 0; i < ILE_OSOBNIKOW; i++)
			Random_osobniki_y[i] = RandomFloat(zakres_min_x, zakres_max_x);

		float *dev_Random_osobniki_y;
		cudaMalloc(&dev_Random_osobniki_y, ILE_OSOBNIKOW * sizeof(float));
		cudaMemcpy(dev_Random_osobniki_y, Random_osobniki_y, ILE_OSOBNIKOW * sizeof(float), cudaMemcpyHostToDevice);

		//Generowanie populacji
		Generuj_Populacje_gpu << <Ilosc_blokow, THREADS_PER_BLOCK >> > (Populacja_gpu_dev, dev_Random_osobniki_x, dev_Random_osobniki_y);

		cudaMemcpy(Populacja_gpu, Populacja_gpu_dev, size, cudaMemcpyDeviceToHost);

		cudaFree(dev_Random_osobniki_x);
		cudaFree(dev_Random_osobniki_y);

		delete[]Random_osobniki_x;
		delete[]Random_osobniki_y;

		if (Czy_wyswietl_generacje == 1)
			cout << endl << "Generacja: 1" << endl << endl;

		if (Czy_wyswietl_osobniki_gpu == 1)
			Wypisz_Populacje(Populacja_gpu);

		//Tworzenie wektorow thrust do przechowywania populacji na host i device
		thrust::host_vector<Osobnik> host_thrust_osobnik(ILE_OSOBNIKOW);
		thrust::device_vector<Osobnik> device_thrust_osobnik(host_thrust_osobnik);
		Osobnik* thrust_pointer = thrust::raw_pointer_cast(&device_thrust_osobnik[0]);	//Wskaznik na miejsce w pamieci na device gdzie przechowywana jest populacja, czyli na poczatek wektoru


		for (int i = 0; i < ILE_OSOBNIKOW; i++)
		{
			host_thrust_osobnik[i] = Populacja_gpu[i];
		}

		if (czy_zapisac_do_pliku == 1)
		{
			for (int i = 0; i < ILE_OSOBNIKOW; i++)
			{
				plik_gpu << host_thrust_osobnik[i].x << "," << host_thrust_osobnik[i].y << "\n";
			}
		}

		//Wektory w ktorych beda przechowywane losowe liczby potrzebne do operacji crossover
		thrust::host_vector<float> czy_crossover_gpu(ILE_OSOBNIKOW);
		thrust::host_vector<float> cross_x_gpu(ILE_OSOBNIKOW);
		thrust::host_vector<float> cross_y_gpu(ILE_OSOBNIKOW);
		thrust::host_vector<float> rodzic1_rand(ILE_OSOBNIKOW);
		thrust::host_vector<float> rodzic2_rand(ILE_OSOBNIKOW);


		//Wektory w ktorych beda przechowywane losowe liczby potrzebne do operacji mutacji
		thrust::host_vector<float> czy_mutacja_gpu(ILE_OSOBNIKOW);
		thrust::host_vector<float> mutation_step_x_gpu(ILE_OSOBNIKOW);
		thrust::host_vector<float> mutation_step_y_gpu(ILE_OSOBNIKOW);


		//Przebieg wielu pokolen populacji
		for (int gen = 0; gen < ILE_GENERACJI - 1; gen++)
		{
			if (Czy_wyswietl_generacje == 1)
				cout << endl << "Generacja: " << gen + +2 << endl << endl;

			//Kopiowanie osobnikow z hosta na device
			device_thrust_osobnik = host_thrust_osobnik;

			if (Czy_wyswietl_osobniki_gpu == 1)
			{
				cout << "Nie policzone" << endl;
				Osobnik* Pointer_do_wypisywania = thrust::raw_pointer_cast(&host_thrust_osobnik[0]);
				Wypisz_Populacje(Pointer_do_wypisywania);
			}

			//Liczenie fitness dla kazdego osobnika w populacji
			Policz_Fitness_gpu << < Ilosc_blokow, THREADS_PER_BLOCK >> > (thrust_pointer);

			if (Czy_wyswietl_osobniki_gpu == 1)
			{
				host_thrust_osobnik = device_thrust_osobnik;		//Kopiowanie populacji z device na hosta

				cout << "Policzone" << endl;
				Osobnik* Pointer_do_wypisywania = thrust::raw_pointer_cast(&host_thrust_osobnik[0]);
				Wypisz_Populacje(Pointer_do_wypisywania);
			}


			//Sortowanie osobnikow w populacji malejaco wedlug fitnessu
			thrust::sort(device_thrust_osobnik.begin(), device_thrust_osobnik.end());

			if (Czy_wyswietl_osobniki_gpu == 1)
			{
				host_thrust_osobnik = device_thrust_osobnik;

				cout << endl << "Posortowane: " << endl;
				Osobnik* Pointer_do_wypisywania = thrust::raw_pointer_cast(&host_thrust_osobnik[0]);
				Wypisz_Populacje(Pointer_do_wypisywania);
			}


			//Przypisywanie osobnikom prawdopodobnienstwa na podstawie rangi wybrania na podstawie rankingu - kolejnosci
			Policz_Prawdopodobienstwo << < Ilosc_blokow, THREADS_PER_BLOCK >> > (thrust_pointer);

			host_thrust_osobnik = device_thrust_osobnik;

			if (Czy_wyswietl_osobniki_gpu == 1)
			{
				cout << "z Policzonym prawdp" << endl;
				Osobnik* Pointer_do_wypisywania = thrust::raw_pointer_cast(&host_thrust_osobnik[0]);
				Wypisz_Populacje(Pointer_do_wypisywania);
			}


			//Crossover

			//Wektory na device w ktorych bedzie przechowywana nowa populacja juz po crossover
			thrust::device_vector<Osobnik> device_thrust_osobnik_nowy(host_thrust_osobnik);
			Osobnik* thrust_pointer_nowy = thrust::raw_pointer_cast(&device_thrust_osobnik_nowy[0]);

			//Generowanie liczb losowych potrzebnych w kernelach
			for (int i = 0; i < ILE_OSOBNIKOW; i++)
			{
				czy_crossover_gpu[i] = ((float)rand()) / (float)RAND_MAX;
				if (czy_crossover_gpu[i] < CROSSOVER_PROBABILITY)
				{
					cross_x_gpu[i] = ((float)rand()) / (float)RAND_MAX;
					cross_y_gpu[i] = ((float)rand()) / (float)RAND_MAX;
					rodzic1_rand[i] = ((float)rand()) / (float)RAND_MAX;
					rodzic2_rand[i] = ((float)rand()) / (float)RAND_MAX;
				}

				czy_mutacja_gpu[i] = ((float)rand()) / (float)RAND_MAX;
				if (czy_mutacja_gpu[i] < MUTATION_PROBABILITY)
				{
					mutation_step_x_gpu[i] = ((float)rand()) / (float)RAND_MAX;
					mutation_step_y_gpu[i] = ((float)rand()) / (float)RAND_MAX;
				}
			}

			//Kopie wektorów losowych na device
			thrust::device_vector<float> device_czy_crossover_gpu(czy_crossover_gpu);
			thrust::device_vector<float> device_cross_x_gpu(cross_x_gpu);
			thrust::device_vector<float> device_cross_y_gpu(cross_y_gpu);
			thrust::device_vector<float> rodzic1_rand_gpu(rodzic1_rand);
			thrust::device_vector<float> rodzic2_rand_gpu(rodzic2_rand);

			thrust::device_vector<float> device_czy_mutacja_gpu(czy_mutacja_gpu);
			thrust::device_vector<float> device_mutation_step_x_gpu(mutation_step_x_gpu);
			thrust::device_vector<float> device_mutation_step_y_gpu(mutation_step_y_gpu);

			//Wskaźniki na kopie wektorów losowych na device
			float* device_czy_crossover_gpu_pointer = thrust::raw_pointer_cast(&device_czy_crossover_gpu[0]);
			float* device_cross_x_gpu_pointer = thrust::raw_pointer_cast(&device_cross_x_gpu[0]);
			float* device_cross_y_gpu_pointer = thrust::raw_pointer_cast(&device_cross_y_gpu[0]);
			float* rodzic1_rand_gpu_pointer = thrust::raw_pointer_cast(&rodzic1_rand_gpu[0]);
			float* rodzic2_rand_gpu_pointer = thrust::raw_pointer_cast(&rodzic2_rand_gpu[0]);

			float* device_czy_mutacja_gpu_pointer = thrust::raw_pointer_cast(&device_czy_mutacja_gpu[0]);
			float* device_mutation_step_x_gpu_pointer = thrust::raw_pointer_cast(&device_mutation_step_x_gpu[0]);
			float* device_mutation_step_y_gpu_pointer = thrust::raw_pointer_cast(&device_mutation_step_y_gpu[0]);

			//Operacja crossover
			Crossover_gpu << < Ilosc_blokow, THREADS_PER_BLOCK >> > (thrust_pointer, thrust_pointer_nowy, device_czy_crossover_gpu_pointer, device_cross_x_gpu_pointer, device_cross_y_gpu_pointer, rodzic1_rand_gpu_pointer, rodzic2_rand_gpu_pointer);

			//Operacja mutacji
			Mutacja_gpu << < Ilosc_blokow, THREADS_PER_BLOCK >> > (thrust_pointer_nowy, device_czy_mutacja_gpu_pointer, device_mutation_step_x_gpu_pointer, device_mutation_step_y_gpu_pointer);

			host_thrust_osobnik = device_thrust_osobnik_nowy;

			if (Czy_wyswietl_osobniki_gpu == 1)
			{
				cout << "Nowe osobniki" << endl;
				Osobnik* Pointer_do_wypisywania = thrust::raw_pointer_cast(&host_thrust_osobnik[0]);
				Wypisz_Populacje(Pointer_do_wypisywania);
			}
			if (czy_zapisac_do_pliku == 1)
			{
				for (int i = 0; i < ILE_OSOBNIKOW; i++)
				{
					plik_gpu << host_thrust_osobnik[i].x << "," << host_thrust_osobnik[i].y << "\n";
				}
			}

		}

		//Znajdywanie optimum

		device_thrust_osobnik = host_thrust_osobnik;
		Policz_Fitness_gpu << < Ilosc_blokow, THREADS_PER_BLOCK >> > (thrust_pointer);
		thrust::sort(device_thrust_osobnik.begin(), device_thrust_osobnik.end());	//Sortowanie
		host_thrust_osobnik = device_thrust_osobnik;

		float x_opt_gpu  = 0;
		float y_opt_gpu = 0;

		for (int i = 0; i < ile_wliczonych; i++)
		{
			x_opt_gpu += host_thrust_osobnik[i].x;
			y_opt_gpu += host_thrust_osobnik[i].y;
		}
		x_opt_gpu /= ile_wliczonych;
		y_opt_gpu /= ile_wliczonych;

		cout << "Optimum w x = " << x_opt_gpu << " y = " << y_opt_gpu << endl;

		Osobnik osobnik_opt_gpu;
		osobnik_opt_gpu.x = x_opt_gpu;
		osobnik_opt_gpu.y = y_opt_gpu;
		osobnik_opt_gpu.fitness = FitFunction(osobnik_opt_gpu);

		cout << "Wartosc funkcji w optimum: " << -osobnik_opt_gpu.fitness << endl;


		if (czy_zapisac_do_pliku == 1)
		{
			plik_gpu << x_opt_gpu << "," << y_opt_gpu << "\n";
			plik_gpu << zakres_min_x << "," << zakres_max_x << "\n" << zakres_min_y << "," << zakres_max_y << "\n";


			if (plik_gpu.good() == true)
			{
				cout << "Dane pomyslnie zapisane do pliku" << endl;
			}
			else
			{
				cout << "Dostep do pliku zostal zabroniony!" << std::endl;
				system("PAUSE");
			}

			plik_gpu.close();
		}

		cudaFree(Populacja_gpu_dev);
		delete[] Populacja_gpu;

		high_resolution_clock::time_point end_time_gpu = high_resolution_clock::now();
		duration_gpu = duration_cast<duration<double>>(end_time_gpu - begin_time_gpu);

		cout << "Czas liczenia na GPU " << duration_gpu.count() << endl << endl;

	}

	if(czy_licz_cpu==1 && czy_licz_gpu==1)
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


void Generuj_Populacje(vector <Osobnik> &Populacja)
{
	for (int i = 0; i < ILE_OSOBNIKOW; i++)
	{
		float x_rand = RandomFloat(zakres_min_x, zakres_max_x);
		float y_rand = RandomFloat(zakres_min_y, zakres_max_y);

		Osobnik osobnik;
		osobnik.x = x_rand;
		osobnik.y = y_rand;
		//osobnik.fitness = 0;
		//osobnik.probability = 0;

		Populacja.push_back(osobnik);
	}

	return;
}


float FitFunction(Osobnik osobnik)
{
	float fittness = 0;
	if (KTORA_FUNKCJA == 1)
		fittness = -(sin(osobnik.x) + cos(osobnik.y));
	else if (KTORA_FUNKCJA == 2)
		fittness = -((osobnik.x)*(osobnik.x) + (osobnik.y)*(osobnik.y));
	else if (KTORA_FUNKCJA == 3)
		fittness = -((osobnik.y)*sin(osobnik.x) - (osobnik.x)*cos(osobnik.y));
	else if (KTORA_FUNKCJA == 4)
		fittness = -(sin(osobnik.x + osobnik.y) + (osobnik.x - osobnik.y)*(osobnik.x - osobnik.y) - 1.5*osobnik.x + 2.5*osobnik.y + 1);
	else if (KTORA_FUNKCJA == 5)
		fittness = -(-(osobnik.y + 47)*sin(sqrt(abs((osobnik.x / 2) + (osobnik.y + 47)))) - osobnik.x*sin(sqrt(abs(osobnik.x - (osobnik.y + 47)))));
	else if (KTORA_FUNKCJA == 6)
		fittness = -(418.9829 * 2 - (osobnik.x*sin(sqrt(abs(osobnik.x))) + osobnik.y*sin(sqrt(abs(osobnik.y)))));
	return fittness;
}


bool compareByFit(const Osobnik &a, const Osobnik &b)
{
	return a.fitness > b.fitness;
}

bool compareByFit_gpu(const Osobnik &a, const Osobnik &b)
{
	return a.fitness > b.fitness;
}

void Wypisz_Populacje(vector <Osobnik> Populacja)
{
	cout << endl;
	for (int j = 0; j < Populacja.size(); j++)
	{
		cout << "Osobnik " << j + 1 << ": x=" << Populacja[j].x << "  y=" << Populacja[j].y << " fit= " << Populacja[j].fitness <<" prob: "<<Populacja[j].probability<< endl;
	}
	cout << endl;
}

void Wypisz_Populacje(Osobnik *Populacja)
{
	cout << endl;
	for (int j = 0; j < ILE_OSOBNIKOW; j++)
	{
		cout << "Osobnik " << j + 1 << ": x=" << Populacja[j].x << "  y=" << Populacja[j].y << " fit= " << Populacja[j].fitness << " prob: " << Populacja[j].probability << endl;
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

int Roulete_Selection(vector <Osobnik> Populacja)
{
	float offset = 0.0;
	float rand_cross = ((float)rand()) / (float)RAND_MAX;
	int rodzic = 0;

	//Losowanie ruletkowe
	for (int j = 0; j < Populacja.size(); j++)
	{
		offset += Populacja[j].probability;
		if (offset > rand_cross)
		{
			rodzic = j;
			break;
		}
	}

	return rodzic;
}
/*
void Mutuj(Osobnik &osobnik)
{
	int znak1 = rand() % 2;
	int znak2 = rand() % 2;

	if (Czy_wyswietl_osobniki_cpu == 1)
		cout << "znak= " << znak1 << " znak2 = " << znak2 << endl;

	float Mutation_step_x = RandomFloat(Mutation_step_min, Mutation_step_max);
	float Mutation_step_y = RandomFloat(Mutation_step_min, Mutation_step_max);

	if (znak1 < 1)
	{
		if (znak2 < 1)
		{
			osobnik.x = osobnik.x - Mutation_step_x;
			if (osobnik.x > zakres_max_x)
				osobnik.x = zakres_max_x;
			if (osobnik.x < zakres_min_x)
				osobnik.x = zakres_min_x;

			osobnik.y = osobnik.y - Mutation_step_y;

			if (osobnik.y > zakres_max_y)
				osobnik.y = zakres_max_y;
			if (osobnik.y < zakres_min_y)
				osobnik.y = zakres_min_y;
		}
		if (znak2 >= 1)
		{
			osobnik.x = osobnik.x - Mutation_step_x;
			if (osobnik.x > zakres_max_x)
				osobnik.x = zakres_max_x;
			if (osobnik.x < zakres_min_x)
				osobnik.x = zakres_min_x;


			osobnik.y = osobnik.y + Mutation_step_y;

			if (osobnik.y > zakres_max_y)
				osobnik.y = zakres_max_y;
			if (osobnik.y < zakres_min_y)
				osobnik.y = zakres_min_y;
		}
	}

	if (znak1 >= 1)
	{
		if (znak2 < 1)
		{
			osobnik.x = osobnik.x + Mutation_step_x;

			if (osobnik.x > zakres_max_x)
				osobnik.x = zakres_max_x;
			if (osobnik.x < zakres_min_x)
				osobnik.x = zakres_min_x;

			osobnik.y = osobnik.y - Mutation_step_y;

			if (osobnik.y > zakres_max_y)
				osobnik.y = zakres_max_y;
			if (osobnik.y < zakres_min_y)
				osobnik.y = zakres_min_y;
		}

		if (znak2 >= 1)
		{
			osobnik.x = osobnik.x + Mutation_step_x;

			if (osobnik.x > zakres_max_x)
				osobnik.x = zakres_max_x;
			if (osobnik.x < zakres_min_x)
				osobnik.x = zakres_min_x;

			osobnik.y = osobnik.y + Mutation_step_y;

			if (osobnik.y > zakres_max_y)
				osobnik.y = zakres_max_y;
			if (osobnik.y < zakres_min_y)
				osobnik.y = zakres_min_y;
		}
	}

	return;
}
*/
void Mutuj(Osobnik &osobnik)
{

	float Mutation_step_x = RandomFloat(-Mutation_step_max, Mutation_step_max);

	/*
	if (abs(Mutation_step_x) < Mutation_step_min)
	{
		if (Mutation_step_x < 0)
			Mutation_step_x = -Mutation_step_min;
		else
			Mutation_step_x = Mutation_step_min;
	}
	*/
	float Mutation_step_y = RandomFloat(-Mutation_step_max, Mutation_step_max);

	/*
	if (abs(Mutation_step_y) < Mutation_step_min)
	{
		if (Mutation_step_y < 0)
			Mutation_step_y = -Mutation_step_min;
		else
			Mutation_step_y = Mutation_step_min;
	}
	*/

	osobnik.x = osobnik.x + Mutation_step_x;
	if (osobnik.x > zakres_max_x)
		osobnik.x = zakres_max_x;
	if (osobnik.x < zakres_min_x)
		osobnik.x = zakres_min_x;

	osobnik.y = osobnik.y + Mutation_step_y;
	if (osobnik.y > zakres_max_y)
		osobnik.y = zakres_max_y;
	if (osobnik.y < zakres_min_y)
		osobnik.y = zakres_min_y;


	return;
}

void Policz_Fitness(vector <Osobnik> &Populacja)
{
	//Obliczanie fitness dla każdego osobnika
	for (int j = 0; j < Populacja.size(); j++)
	{
		Populacja[j].fitness = FitFunction(Populacja[j]);
	}

	return;
}

