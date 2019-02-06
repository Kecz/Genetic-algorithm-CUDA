# Genetic-algorithm-CUDA
This project is a genetic algorithm running on GPU using CUDA C/C++.

Algorithm is based on:
- Rank base fitness estimate
- Roulette selection
- Blend Crossover for Real-Coded Genetic Algorithms

To use algorithm:  
  kernel.cu

There are 6 build-in functions to test algorithm, they are described at the beggining of kernel.cu. To change selected function adjust parameter 'KTORA_FUNKCJA'.

Running kernel.cu shows in console value of finded optimum and creates csv file with the course of each generation.

To plot course of each generation use matlab script:  
  plot_data.m
  
This script provides 2D and 3D visualisations of each generation on selected surface.
Selected function, min range and max range of searching are saved in csv file and read automatically in plot_data.m but you may need to change the resolution of plotted functions with parameter 'podzialka'.
 
