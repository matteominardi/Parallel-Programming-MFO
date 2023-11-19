#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NUM_DIMENSIONS 5     // Define the number of dimensions
#define POPULATION_SIZE 3     // Define the population size
#define LOWER_BOUND -5.0      // Lower bound for the search space
#define UPPER_BOUND 5.0       // Upper bound for the search space
#define MAX_ITERATIONS 1000     // Maximum number of iterations
#define BETA_INIT 2.0          // Initial value of beta
#define ALPHA 1.2              // Constant alpha for moth movement
#define EARLY_STOPPING_PATIENCE 100 // Number of iterations to wait before early stopping

#define M_PI 3.14159265358979323846 // PI

// Function to optimize (e.g., Sphere function)
double sphereFunction(double *position) {
    double result = 0.0;
    int i = 0;
    for (i = 0; i < NUM_DIMENSIONS; ++i) {
        result += position[i] * position[i];
    }
    return result;
}

double Quadric(double *position) {
	unsigned int i, j;
	double fitaux, fitness = 0.0;

	for (i = 1; i <= NUM_DIMENSIONS; ++i) {
		fitaux = 0;

		for (j = 0; j < i; ++j) {
			fitaux += (double) position[j];
		}

		fitness += fitaux * fitaux;
	}
	return fitness;
}

double fitness_function(double *position) {
    return sphereFunction(position);
}

// Function to initialize a random position
void initializePosition(double *position) {
    int i = 0;
    for (i = 0; i < NUM_DIMENSIONS; ++i) {
        position[i] = ((double)rand() / RAND_MAX) * (UPPER_BOUND - LOWER_BOUND) + LOWER_BOUND;
    }
}

float n_flames_update(float n_fm, unsigned curr_iter) {
    return round(n_fm - curr_iter * (n_fm - 1) / MAX_ITERATIONS);
}

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

// Structure to hold both value and its original index
typedef struct {
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

int main() {
    srand(time(NULL));

    // Initialize population
    double population[POPULATION_SIZE][NUM_DIMENSIONS];
    double flames[POPULATION_SIZE][NUM_DIMENSIONS];
    double fitness[POPULATION_SIZE];
    double flameFitness[POPULATION_SIZE];
    float n_flames = (float) POPULATION_SIZE;
    unsigned early_stopping_counter = 0;
    double previous_best_fitness = 0.0;

    int i = 0;
    for (i = 0; i < POPULATION_SIZE; ++i) {
        initializePosition(population[i]);
        fitness[i] = fitness_function(population[i]);
        int j = 0;
        for (j = 0; j < NUM_DIMENSIONS; ++j) {
            flames[i][j] = population[i][j];
            flameFitness[i] = fitness[i];
        }
    }

    double beta = BETA_INIT;
    unsigned iteration = 0;
    for (iteration = 0; iteration < MAX_ITERATIONS; ++iteration) {
        int i = 0;
        for (i = 0; i < POPULATION_SIZE; ++i) {
            int j = 0;
            for (j = 0; j < NUM_DIMENSIONS; ++j) {
                population[i][j] = population[i][j] + beta * (flames[i][j] - population[i][j]) + ALPHA * ((double)rand() / RAND_MAX - 0.5);
                // population[i][j] = population[i][j] + beta * (flames[i][j] - population[i][j]) * cos((double)(2 * log(beta) * M_PI));
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
                flameFitness[i] = currentFitness;
            }
        }

        int *sorted_indexes = argsort(flameFitness, POPULATION_SIZE);
        double minFlameFitness = flameFitness[sorted_indexes[0]];

        if (iteration == 0) {
            previous_best_fitness = minFlameFitness;
        } else {
            if (minFlameFitness < previous_best_fitness) {
                previous_best_fitness = minFlameFitness;
                early_stopping_counter = 0;
            } else {
                early_stopping_counter++;
            }
        }

        // Updating the number of flames at each iteration seems to be exploiting too much
        // Commenting seems to be allowing more exploration
        // if (POPULATION_SIZE > 1) {
        //     int j = 0;
        //     n_flames = n_flames_update(n_flames, iteration);
        //     printf("n_flames = %f\n", n_flames);
        //     int i = (int) n_flames;
        //     for (i = (int) n_flames; i < POPULATION_SIZE; ++i) {
        //         for (j = 0; j < NUM_DIMENSIONS; ++j) {
        //             flames[sorted_indexes[i]][j] = flames[sorted_indexes[i-1]][j];
        //         }
        //     }
        // }

        beta = BETA_INIT * my_exp(-iteration / (double)MAX_ITERATIONS); // Change iteration to double for accurate division
        // beta = BETA_INIT * my_exp(((-iteration / (double)MAX_ITERATIONS) - 1) * rand() + 1); // Update of old version in mfo.c seems to not be working

        if (iteration % 10 == 0)
            printf("Iteration %d: Best Fitness Value = %lf\n", iteration, minFlameFitness);

        if (early_stopping_counter >= EARLY_STOPPING_PATIENCE) {
            printf("Early stopping at iteration %d.\n", iteration);
            break;
        }
    }

    printf("Best solution found = %lf\n", previous_best_fitness);

    return 0;
}
