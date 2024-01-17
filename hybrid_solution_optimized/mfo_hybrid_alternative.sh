#!/bin/bash

#PBS -l select=2:ncpus=16:mem=32gb -l place=pack:excl
#PBS -l walltime=00:10:00
#PBS -q short_cpuQ

module load mpich-3.2
cd $PBS_O_WORKDIR
mpicc -g -Wall -fopenmp -lm -o mfo_hybrid_alternative mfo_hybrid_alternative.c
mpirun.actual -n 8 ./mfo_hybrid_alternative