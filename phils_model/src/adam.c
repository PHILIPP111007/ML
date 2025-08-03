#include <stdlib.h>
#include <math.h>
#include "functions.h"
#include "adam.h"

///////////////////////////////////////////////////////////////////////////////
// Adam
///////////////////////////////////////////////////////////////////////////////

// Freeing up optimizer resources
void destroy_adam(struct AdamOptimizer *opt, float *layer_sizes, int layer_sizes_rows, int layer_sizes_cols) {
    for (int layer_index = 0; layer_index < layer_sizes_rows; layer_index++) {
        int n_inputs = (int)layer_sizes[layer_index * layer_sizes_cols + 0];

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

    for (int layer_index = 0; layer_index < layer_sizes_rows; layer_index++) {
        const int n_inputs = (int)layer_sizes[layer_index * layer_sizes_cols];
        const int n_neurons = (int)layer_sizes[layer_index * layer_sizes_cols + 1];

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

void adam_step(struct AdamOptimizer *optimizer, float ***weights, float ***grads, float *layer_sizes, int layer_sizes_rows, int layer_sizes_cols, float max_change) {
    const float b1 = optimizer->b1;
    const float b2 = optimizer->b2;
    const float lr = optimizer->lr;
    const float eps = optimizer->eps;
    const int epoch = ++optimizer->epoch;
    const float b1_pow = powf(b1, epoch);
    const float b2_pow = powf(b2, epoch);
    const float inv_1mb1 = 1.0f / (1.0f - b1_pow);
    const float inv_1mb2 = 1.0f / (1.0f - b2_pow);
    const float b1_minus_1 = 1.0f - b1;
    const float b2_minus_1 = 1.0f - b2;

    for (int layer_index = 0; layer_index < layer_sizes_rows; layer_index++) {
        const int n_inputs = (int)layer_sizes[layer_index * layer_sizes_cols];
        const int n_neurons = (int)layer_sizes[layer_index * layer_sizes_cols + 1];

        float **layer_weights = weights[layer_index];
        float **layer_grads = grads[layer_index];
        float **layer_m = optimizer->m[layer_index];
        float **layer_v = optimizer->v[layer_index];

        #pragma omp parallel for collapse(2)
        for (int i = 0; i < n_inputs; i++) {
            for (int j = 0; j < n_neurons; j++) {
                const float grad = layer_grads[i][j];
                const float grad_sq = grad * grad;

                // Update moments
                layer_m[i][j] = b1 * layer_m[i][j] + b1_minus_1 * grad;
                layer_v[i][j] = b2 * layer_v[i][j] + b2_minus_1 * grad_sq;

                // Compute bias-corrected moments
                const float m_hat = layer_m[i][j] * inv_1mb1;
                const float v_hat = layer_v[i][j] * inv_1mb2;

                // Update weights
                float delta = lr * m_hat / (sqrtf(v_hat) + eps);
                layer_weights[i][j] -= safe_update(delta, max_change);

                // Handle NaN
                layer_weights[i][j] = isnan(layer_weights[i][j]) ? 0.0f : layer_weights[i][j];
            }
        }
    }
}
