#include <stdlib.h>
#include <math.h>
#include "init.h"


double *init_bias(int n_neurons, int n_inputs) {
    double* bias = malloc(n_neurons * sizeof(double));
    double std_dev = sqrtf(2.0 / (n_neurons + n_inputs));

    for (int i = 0; i < n_neurons; ++i) {
        bias[i] = ((double)rand() / RAND_MAX) * std_dev * sqrtf(2.0 / (n_neurons + n_inputs)) * 2.0 - std_dev * sqrtf(2.0 / (n_neurons + n_inputs));
        bias[i] *= 100.0;
    }
    return bias;
}

double **init_weights(int n_neurons, int n_inputs) {
    double **weights = malloc(n_inputs * sizeof(double*));
    for (int i = 0; i < n_inputs; i++) {
        weights[i] = malloc(n_neurons * sizeof(double));
    }

    double std_dev = sqrtf(2.0 / (n_neurons + n_inputs));

    for (int i = 0; i < n_inputs; i++) {
        for (int j = 0; j < n_neurons; j++) {
            weights[i][j] = ((double)rand() / RAND_MAX) * std_dev * sqrtf(2.0 / (n_neurons + n_inputs)) * 2.0 - std_dev * sqrtf(2.0 / (n_neurons + n_inputs));
            weights[i][j] *= 100.0;
        }
    }
    return weights;
}
