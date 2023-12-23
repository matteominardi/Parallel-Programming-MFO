#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NUM_DIMENSIONS 200000           // Define the number of dimensions (15 best)
#define POPULATION_SIZE 210           // Define the population size (2 best) 
#define FIX_N_FLAMES 1              // Fix the number of flames or not (0 = no, 1 = yes) 
#define LOWER_BOUND -20.0            // Lower bound for the search space
#define UPPER_BOUND 20.0             // Upper bound for the search space
#define MAX_ITERATIONS 50           // Maximum number of iterations
#define BETA_INIT 1.0               // Initial value of beta
#define ALPHA 0.2                   // Constant alpha for moth movement
#define EARLY_STOPPING 0            // Early stopping or not (0 = no, 1 = yes)
#define EARLY_STOPPING_PATIENCE 3 // Number of iterations to wait before early stopping

#define M_PI 3.14159265358979323846 // PI

// Benchmark functions prototypes
double Sphere(double *position);
double Quadric(double *position);
double Hyperellipsoid(double *position);
double Rastrigin(double *position);
double Griewank(double *position);
double Weierstrass(double *position);
double Ackley(double *position);

// Chosen function
double fitness_function(double *position) {
    return Ackley(position);
}

// Helper functions prototypes
double my_exp(double x);
int compareIndexedValue(const void *a, const void *b);
int* argsort(double arr[], int size);
void print_solution(double *position);

// MFO functions
void initialize_position(double *position) {
    int i = 0;
    for (i = 0; i < NUM_DIMENSIONS; ++i) {
        position[i] = ((double)rand() / RAND_MAX) * (UPPER_BOUND - LOWER_BOUND) + LOWER_BOUND;
    }
}

float n_flames_update(float n_fm, unsigned curr_iter) {
    return round(n_fm - curr_iter * (n_fm - 1) / MAX_ITERATIONS);
}

double population_update(double previous, double beta, double flame) {
    return previous + beta * (flame - previous) + ALPHA * ((double)rand() / RAND_MAX - 0.5);
    // return previous + beta * (flame - previous) * cos((double)(2 * log(beta) * M_PI)
}

double beta_update(int iteration) {
    return BETA_INIT * exp(-iteration / (double)MAX_ITERATIONS); // Change iteration to double for accurate division
    // return BETA_INIT * my_exp(((-iteration / (double)MAX_ITERATIONS) - 1) * rand() + 1); // Update of old version in mfo.c seems to not be working
}

int main() {
    srand(time(NULL));

    MPI_Init(NULL, NULL); 

    double start, finish;
    start = MPI_Wtime();
 
    // Initialize population
    double population[POPULATION_SIZE][NUM_DIMENSIONS];
    double flames[POPULATION_SIZE][NUM_DIMENSIONS];
    double fitness[POPULATION_SIZE];
    double flames_fitness[POPULATION_SIZE];
    float n_flames = (float) POPULATION_SIZE;
    unsigned early_stopping_counter = 0;
    double *best_solution;
    double best_fitness = 0.0;

    int i = 0;
    for (i = 0; i < POPULATION_SIZE; ++i) {
        initialize_position(population[i]);
        fitness[i] = fitness_function(population[i]);

        int j = 0;
        for (j = 0; j < NUM_DIMENSIONS; ++j) {
            flames[i][j] = population[i][j];
            flames_fitness[i] = fitness[i];
        }
    }

    double beta = BETA_INIT;

    int iteration = 0;
    for (iteration = 0; iteration < MAX_ITERATIONS; ++iteration) {
        int i = 0;
        for (i = 0; i < POPULATION_SIZE; ++i) {
            int j = 0;
            for (j = 0; j < NUM_DIMENSIONS; ++j) {
                population[i][j] = population_update(population[i][j], beta, flames[i][j]);
                
                if (population[i][j] < LOWER_BOUND)
                    population[i][j] = LOWER_BOUND;
                else if (population[i][j] > UPPER_BOUND)
                    population[i][j] = UPPER_BOUND;
            }

            double currentFitness = fitness_function(population[i]);
            if (currentFitness < fitness[i]) {
                fitness[i] = currentFitness;
                int j = 0;
                for (j = 0; j < NUM_DIMENSIONS; ++j)
                    flames[i][j] = population[i][j];
                flames_fitness[i] = currentFitness;
            }
        }

        double min_flame_fitness = fitness_function(flames[0]);
        best_solution = flames[0];
        i = 1;
        for (i = 1; i < POPULATION_SIZE; ++i) {
            double current_fitness = fitness_function(flames[i]);
            
            if (current_fitness < min_flame_fitness) {
                min_flame_fitness = current_fitness;
                best_solution = flames[i];
            } 
        }

        if (iteration == 0) {
            best_fitness = min_flame_fitness;
        } else {
            if (min_flame_fitness < best_fitness) {
                best_fitness = min_flame_fitness;

                if (EARLY_STOPPING == 1) {
                    early_stopping_counter = 0;
                }
            } else {
                if (EARLY_STOPPING == 1) {
                    early_stopping_counter++;
                }
            }
        }

        printf("Iteration %d: Best Fitness Value = %lf\n", iteration + 1, min_flame_fitness);

        if (EARLY_STOPPING == 1 && early_stopping_counter >= EARLY_STOPPING_PATIENCE) {
            printf("Early stopping at iteration %d.\n", iteration);
            break;
        }

        if (FIX_N_FLAMES == 0) {
            if (POPULATION_SIZE > 1) {
                int *sorted_indexes = argsort(flames_fitness, POPULATION_SIZE);
                double n_flames_prev = n_flames;
                n_flames = n_flames_update(n_flames, iteration);
                
                if (n_flames != n_flames_prev) {
                    printf("Number of flames changed: n_flames = %d\n", (int)n_flames);
                }

                int i = (int) n_flames;
                for (i = (int) n_flames; i < POPULATION_SIZE; ++i) {
                    int j = 0;
                    for (j = 0; j < NUM_DIMENSIONS; ++j) {
                        flames[sorted_indexes[i]][j] = flames[sorted_indexes[i-1]][j];
                    }
                }
            }
        }

        beta = beta_update(iteration);
    }
    finish = MPI_Wtime();

    printf("\n\n");
    printf("\nBest fitness = %lf\n", best_fitness);    
    printf("\n\nTime elapsed: %lf", finish - start);

    MPI_Finalize();

    return 0;
}

// Benchmark functions
double Sphere(double *position) {
    double result = 0.0;

    int i = 0;
    for (i = 0; i < NUM_DIMENSIONS; ++i) {
        result += position[i] * position[i];
    }

    return result;
}

double Quadric(double *position) {
	double fitaux, fitness = 0.0;

	int i = 1;
	for (i = 1; i <= NUM_DIMENSIONS; ++i) {
		fitaux = 0;

        int j = 0;
		for (j = 0; j < i; ++j) {
			fitaux += (double) position[j];
		}

		fitness += fitaux * fitaux;
	}

	return fitness;
}

double Hyperellipsoid(double *position) {
    double fitness = 0.0;

	int i;
	for (i = 0; i < NUM_DIMENSIONS; ++i) {
		fitness += i * (double) (position[i] * position[i]);
	}

	return fitness;
}

double Rastrigin(double *position) {
	double fitness = 0.0;

	int i = 0;
	for (i = 0; i < NUM_DIMENSIONS; ++i) {
		fitness += (position[i] * position[i] - 10 *
			cos(2.0 * M_PI * position[i]) + 10);
	}

	return fitness;
}

double Griewank(double *position) {
	double fitness1 = 0.0, fitness2 = 1.0, fitness = 0.0;

	int i = 0;
	for (i = 0; i < NUM_DIMENSIONS; ++i) {
		fitness1 += (double) (position[i] * position[i]);
		fitness2 *= cos(position[i] / sqrt(i + 1.0));
	}

	fitness = 1 + (fitness1 / 4000) - fitness2;
	return fitness;
}

double Weierstrass(double *position) {
	double res;
	double sum;
	double a, b;
	unsigned int k_max;

	a = 0.5;
	b = 3.0;
	k_max = 20;
	res = 0.0;

	int i = 0;
	for (i = 0; i < NUM_DIMENSIONS; ++i) {
		sum = 0.0;

        int j = 0;
		for (j = 0; j <= k_max; j++)
			sum += pow(a, j) *
				cos(2.0 * M_PI * pow(b, j) *
				(position[i] + 0.5));

		res += sum;
	}

	sum = 0.0;

    int j = 0;
	for (j = 0; j <= k_max; ++j) {
		sum += pow(a,j) * cos(2.0 * M_PI * pow(b, j) * 0.5);
	}

	return res - NUM_DIMENSIONS * sum;
}

double Ackley(double *position) {
    double e = 2.71828182845904523536;  // Euler's number
	double fitaux1 = 0.0, fitaux2 = 0.0, fitness = 0.0;
	
	int j = 0;
	for (j = 0; j < NUM_DIMENSIONS; ++j) {
		fitaux1 += position[j] * position[j];
		fitaux2 += cos(2 * M_PI * position[j]);
	}

	fitness = -20 * exp(-0.2 * sqrt(fitaux1 / NUM_DIMENSIONS))
		- exp(fitaux2 / NUM_DIMENSIONS) + 20 + e;

	return fitness;
}

// Helper functions
double my_exp(double x) {
    double result = 1.0;
    double term = 1.0;

    int i = 1;
    for (i = 1; i < 20; ++i) { // Adjust the number of iterations for desired precision
        term *= x / i;
        result += term;
    }

    return result;
}

typedef struct { // Structure to hold both value and its original index
    int value;
    int index;
} IndexedValue;

// Function to compare IndexedValue structures
int compareIndexedValue(const void *a, const void *b) {
    IndexedValue *x = (IndexedValue *)a;
    IndexedValue *y = (IndexedValue *)b;

    return x->value - y->value;
}

// Function to perform argsort and return sorted indexes
int* argsort(double arr[], int size) {
    IndexedValue *indexedArr = (IndexedValue *)malloc(size * sizeof(IndexedValue));

    if (indexedArr == NULL) {
        printf("Memory allocation failed.\n");
        exit(EXIT_FAILURE);
    }

    // Populate indexed array with values and their original indexes
    int i = 0;
    for (i = 0; i < size; i++) {
        indexedArr[i].value = arr[i];
        indexedArr[i].index = i;
    }

    // Sort the indexed array based on values while preserving original indexes
    qsort(indexedArr, size, sizeof(IndexedValue), compareIndexedValue);

    // Create an array to store sorted indexes
    int *sortedIndexes = (int *)malloc(size * sizeof(int));
    if (sortedIndexes == NULL) {
        printf("Memory allocation failed.\n");
        exit(EXIT_FAILURE);
    }

    // Populate the sorted indexes based on the sorted indexed array
    i = 0;
    for (i = 0; i < size; i++) {
        sortedIndexes[i] = indexedArr[i].index;
    }

    // Free dynamically allocated memory
    free(indexedArr);

    return sortedIndexes;
}

void print_solution(double *position) {
    int i = 0;
    printf("\t[");
    for (i = 0; i < NUM_DIMENSIONS; ++i) {
        printf("%lf, ", position[i]);
    }
    printf("]\n");
}