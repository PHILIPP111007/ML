#include <stdlib.h>
#include <math.h>
#include "functions.h"
#include "init.h"


float *init_bias(int n_neurons, int n_inputs) {
    float *bias = malloc(n_neurons * sizeof(float));
    check_if_null((float *)bias, "bias");
    float std_dev = sqrtf(2.0 / (n_neurons + n_inputs));

    for (int i = 0; i < n_neurons; i++) {
        bias[i] = (float)((double)rand() / RAND_MAX) * std_dev * sqrtf(2.0 / (n_neurons + n_inputs)) * 2.0 - std_dev * sqrtf(2.0 / (n_neurons + n_inputs));
        bias[i] *= 100.0;
    }
    return bias;
}

float **init_weights(int n_neurons, int n_inputs) {
    float **weights = create_matrix(n_inputs, n_neurons);

    float std_dev = sqrtf(2.0 / (n_neurons + n_inputs));
    for (register int i = 0; i < n_inputs; i++) {
        for (register int j = 0; j < n_neurons; j++) {
            weights[i][j] = (float)((double)rand() / RAND_MAX) * std_dev * sqrtf(2.0 / (n_neurons + n_inputs)) * 2.0 - std_dev * sqrtf(2.0 / (n_neurons + n_inputs));
            weights[i][j] *= 100.0;
        }
    }
    return weights;
}
