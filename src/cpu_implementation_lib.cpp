#include "cpu_implementation_lib.hpp"



void Generate_Population(vector <Member> &Population)
{
	for (int i = 0; i < How_many_members; i++)
	{
		float x_rand = RandomFloat(range_min_x, range_max_x);
		float y_rand = RandomFloat(range_min_y, range_max_y);

		Member member;
		member.x = x_rand;
		member.y = y_rand;


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

void Show_Population(vector <Member> Population)
{
	cout << endl;
	for (int j = 0; j < Population.size(); j++)
	{
		cout << "Member " << j + 1 << ": x=" << Population[j].x << "  y=" << Population[j].y << " fit= " << Population[j].fitness <<" prob: "<<Population[j].probability<< endl;
	}
	cout << endl;
}

void Show_Population(Member *Population)
{
	cout << endl;
	for (int j = 0; j < How_many_members; j++)
	{
		cout << "Member " << j + 1 << ": x=" << Population[j].x << "  y=" << Population[j].y << " fit= " << Population[j].fitness << " prob: " << Population[j].probability << endl;
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

	//Roulette selection
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

void Mutate(Member &member, float mutation_step_maximum)
{

	float Mutation_step_x = RandomFloat(-mutation_step_maximum, mutation_step_maximum);
	float Mutation_step_y = RandomFloat(-mutation_step_maximum, mutation_step_maximum);

	// member.x = member.x + Mutation_step_x;
	if (member.x + Mutation_step_x > range_max_x)
	{
		member.x = range_max_x - (Mutation_step_x - (range_max_x - member.x));
	}
	else if (member.x + Mutation_step_x < range_min_x)
	{
		member.x = range_min_x - (Mutation_step_x - (range_min_x - member.x));
	}
	else 
	{
		member.x = member.x + Mutation_step_x;
	}

	// member.y = member.y + Mutation_step_y;
	if (member.y + Mutation_step_y > range_max_y)
	{
		member.y = range_max_y - (Mutation_step_y - (range_max_y - member.y));
	}
	else if (member.y + Mutation_step_y < range_min_y)
	{
		member.y = range_min_y - (Mutation_step_y - (range_min_y - member.y));
	}
	else
	{
		member.y = member.y + Mutation_step_y;
	}

	return;
}

void Count_Fitness(vector <Member> &Population)
{
	//Counting fitness for every member 
	for (int j = 0; j < Population.size(); j++)
	{
		Population[j].fitness = FitFunction(Population[j]);
	}

	return;
}

void PauseSystem()
{
	cout << "Press enter to continue ...";
	cin.get();
	return;
}
