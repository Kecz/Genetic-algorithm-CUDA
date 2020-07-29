filename_results_cpu = "results/algorithm_results_cpu.csv"
filename_results_gpu = "results/algorithm_results_gpu.csv"
objdir = obj
srcdir = src
objects = $(objdir)/cpu_implementation_lib.o $(objdir)/gpu_implementation_lib.o $(objdir)/config.o $(objdir)/genetic_algorithm.o

default: compile

all: clean compile run

cr: compile run

cc: clean compile

compile: $(objects)
	@echo "Compiling program ..."
	nvcc $(objects) -o run_algorithm.o 
	
run:
	./run_algorithm.o $(filename_results_cpu) $(filename_results_gpu)

debug:	$(objects)
	nvcc -g -G $(objects) -o run_algorithm_debug.o
	cuda-gdb run_algorithm_debug.o

$(objdir)/cpu_implementation_lib.o: $(srcdir)/cpu_implementation_lib.cpp
	nvcc -x cu -I. -dc $< -o $@

$(objdir)/gpu_implementation_lib.o: $(srcdir)/gpu_implementation_lib.cu
	nvcc -I. -dc $< -o $@

$(objdir)/config.o: config.h
	nvcc -x cu -I. -dc $< -o $@

$(objdir)/genetic_algorithm.o: genetic_algorithm.cu
	nvcc -I. -dc $< -o $@

clean:
	@echo "Cleaning ..."
	rm -f run_algorithm.o
	rm -f run_algorithm_debug.o
	rm -f $(objdir)/cpu_implementation_lib.o
	rm -f $(objdir)/gpu_implementation_lib.o
	rm -f $(objdir)/genetic_algorithm.o
	rm -f $(filename_results_cpu)
	rm -f $(filename_results_gpu)

