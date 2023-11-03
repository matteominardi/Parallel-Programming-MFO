#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

const double M_PI = 3.14159265358979323846; // PI

const unsigned n = 2; // problem dimension
const unsigned N = 20; // POP_SIZE
const unsigned N_FM = 20; // number of flames to preserve at each iteration
const unsigned MAX_ITER = 100;

float* init_vector(){
    return (float *) malloc(n * sizeof(float));
}

float** init_matrix() {
    float **matrix = (float **) malloc(N * sizeof(float *));
    
    unsigned i = 0;
    for (i = 0; i < N; i++)
        matrix[i] = init_vector();

    return matrix;
}

// returns a random number between 0 and 1
double random_number() {
    srand((unsigned)time(NULL));
    return (double)rand() / RAND_MAX;
}

float* vector_vector_sum(float *v1, float *v2) {
    float *res = init_vector();
    int i = 0;

    for (i = 0; i < n; i++)
        res[i] = v1[i] + v2[i];

    return res;
}

float* vector_constant_sum(float *v, float c) {
    int i = 0;

    for (i = 0; i < n; i++)
        v[i] = v[i] + c;
}

float delta_distance(float *moth, float *flame) {
    float sum = 0;
    int j = 0;

    for (j = 0; j < n; j ++)
        sum += fabs(moth[j] - flame[j]);
    
    return sum;
}

float t_update(unsigned curr_iter) {
    float r = -1 + curr_iter * (-1.0 / MAX_ITER);
    return (r - 1) * random_number() + 1;
}

float* moth_movement(float **X, float **FM, unsigned N_FM, unsigned i) {
    float b = 1.0;
    float t = t_update(i);
    float *movement = init_vector();
    float offset = 0;
    offset = delta_distance(X[i], FM[i]) * exp(b * t) * cos((double)(2 * t * M_PI));

    if (i <= N_FM)
        movement = vector_vector_sum(movement, FM[i]);
    else
        movement = vector_vector_sum(movement, FM[N_FM]);
    
    movement = vector_constant_sum(movement, offset);

    return movement;
}

unsigned n_fm_update(float n_fm, unsigned curr_iter) {
    return round(n_fm - curr_iter * (n_fm - 1) / MAX_ITER);
}

int main() {
    float **X = init_matrix();
    float **FM = init_matrix();
    float *Fit_X = init_vector();
    float *Fit_FM = init_vector();
    float n_fm = N_FM;
    unsigned curr_iter = 0;

    return 0;
}
