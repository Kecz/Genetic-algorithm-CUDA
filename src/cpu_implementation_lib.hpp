#ifndef cpu_implementation_lib_hpp
#define cpu_implementation_lib_hpp

#include <vector>
#include <iostream>
#include <cmath>

#include "../config.h"
#include "structs_lib.cpp"

using namespace std;

void Generate_Population(vector <Member> &);
float FitFunction(Member);
bool compareByFit(const Member &, const Member &);
void Show_Population(vector <Member>);
void Show_Population(Member *);
float RandomFloat(float, float);
int Roulete_Selection(vector <Member>);
void Mutate(Member &, float);
void Count_Fitness(vector <Member> &);
void PauseSystem();

#endif