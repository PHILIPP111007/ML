#include <stdlib.h>
#include <math.h>
#include "adam.h"

///////////////////////////////////////////////////////////////////////////////
// Adam
///////////////////////////////////////////////////////////////////////////////

// Freeing up optimizer resources
void destroy_adam(struct AdamOptimizer *opt, float *layer_sizes, int layer_sizes_rows, int layer_sizes_cols) {
    for (int layer_index = 0; layer_index < layer_sizes_rows; layer_index++) {
        float n_inputs_float = layer_sizes[layer_index * layer_sizes_cols + 0];
        int n_inputs = (int)n_inputs_float;

        for (int i = 0; i < n_inputs; i++) {
            free(opt->m[layer_index][i]);
            free(opt->v[layer_index][i]);
        }
        free(opt->m[layer_index]);
        free(opt->v[layer_index]);
    }
    free(opt);
}

// Creating a new instance of the optimizer
struct AdamOptimizer *create_adam(float lr, float b1, float b2, float eps, float *layer_sizes, int layer_sizes_rows, int layer_sizes_cols) {
    struct AdamOptimizer *optimizer = malloc(sizeof(struct AdamOptimizer));
    optimizer->lr = lr;
    optimizer->b1 = b1;
    optimizer->b2 = b2;
    optimizer->eps = eps;
    optimizer->epoch = 0;

    float ***m = malloc(layer_sizes_rows * sizeof(float**));
    float ***v = malloc(layer_sizes_rows * sizeof(float**));

    for (int layer_index = 0; layer_index < layer_sizes_rows; ++layer_index) {
        float n_inputs_float = layer_sizes[layer_index * layer_sizes_cols + 0];
        float n_neurons_float = layer_sizes[layer_index * layer_sizes_cols + 1];
        int n_inputs = (int)n_inputs_float;
        int n_neurons = (int)n_neurons_float;

        m[layer_index] =  malloc(n_inputs * sizeof(float*));
        v[layer_index] =  malloc(n_inputs * sizeof(float*));

        for (int i = 0; i < n_inputs; ++i) {
            m[layer_index][i] =  malloc(n_neurons * sizeof(float*));
            v[layer_index][i] =  malloc(n_neurons * sizeof(float*));
        }
    }
    optimizer->m = m;
    optimizer->v = v;

    return optimizer;
}

void adam_step(struct AdamOptimizer *optimizer, float ***weights, float ***grads, float *layer_sizes, int layer_sizes_rows, int layer_sizes_cols) {
    optimizer->epoch++;

    for(int layer_index = 0; layer_index < layer_sizes_rows; layer_index++) {
        float n_inputs_float = layer_sizes[layer_index * layer_sizes_cols + 0];
        float n_neurons_float = layer_sizes[layer_index * layer_sizes_cols + 1];
        int n_inputs = (int)n_inputs_float;
        int n_neurons = (int)n_neurons_float;

        for (int i = 0; i < n_inputs; i++) {
            for (int j = 0; j < n_neurons; j++) {
                // Update the first moment (average of gradients)
                optimizer->m[layer_index][i][j] = optimizer->b1 * optimizer->m[layer_index][i][j] + (1 - optimizer->b1) * grads[layer_index][i][j];
                
                // Update the second moment (mean square of gradients)
                optimizer->v[layer_index][i][j] = optimizer->b2 * optimizer->v[layer_index][i][j] + (1 - optimizer->b2) * pow(grads[layer_index][i][j], 2);
                
                // Correcting the bias
                float m_hat = optimizer->m[layer_index][i][j] / (1 - pow(optimizer->b1, optimizer->epoch));
                float v_hat = optimizer->v[layer_index][i][j] / (1 - pow(optimizer->b2, optimizer->epoch));
                
                // Updating weights
                weights[layer_index][i][j] -= optimizer->lr * m_hat / (sqrt(v_hat) + optimizer->eps);

                if (isnan(weights[layer_index][i][j])) {
                    weights[layer_index][i][j] = 0.0;
                }
            }
        }
    }
}
