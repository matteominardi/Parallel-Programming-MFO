#!/bin/bash

#PBS -l select=1:ncpus=8:mem=4gb -l place=pack:excl
#PBS -l walltime=00:10:00
#PBS -q short_cpuQ

module load mpich-3.2
cd $PBS_O_WORKDIR
mpicc -g -Wall -fopenmp -lm -o mfo_hybrid_best mfo_hybrid_best.c
mpirun.actual -n 8 ./mfo_hybrid_best