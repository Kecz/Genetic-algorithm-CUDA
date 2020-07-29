# Genetic-algorithm-CUDA
This project is a genetic algorithm finding global minimum of a chosen function. It runs on CPU and on GPU using CUDA C/C++.

### Algorithm is based on:
- Rank base fitness estimate
- Roulette selection
- Blend Crossover for Real-Coded Genetic Algorithms

Detailed description of algorithm can be found in the [genetic_algorithm.cu file](genetic_algorithm.cu).

## To compile

        make
or

        make compile

Compiled files will be stored in directory 'obj'.

## To run

        make run

If proper parameters in [config file](config.h) are properly set, results of the algorithm will be saved to csv files in directory 'results'.

## To clean

        make clean

## To clean, compile and run

        make all

## To use nvidia profiler

        sudo nvvp

There are 6 build-in functions to test algorithm, they are described at the beggining of [genetic_algorithm.cu file](genetic_algorithm.cu). To change selected function, adjust parameter 'WHICH_FUNCTION' in [config file](config.h).

Running [genetic_algorithm.cu file](genetic_algorithm.cu) shows in console value of finded optimum and creates csv file with the course of each generation.

## To plot course of each generation use matlab script

        plot_data.m

This script provides 2D and 3D visualisations of each generation on selected surface.
Selected function, min range and max range of searching are saved in csv file and read automatically in plot_data.m but you may need to change the resolution of plotted functions with parameter 'resolution'.  
