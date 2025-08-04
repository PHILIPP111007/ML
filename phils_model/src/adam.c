#include <stdlib.h>
#include <math.h>
#include "functions.h"
#include "adam.h"


inline float fast_pow(float a, int b) {
    return expf(b * logf(a));
}

///////////////////////////////////////////////////////////////////////////////
// Adam optimizer
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

inline void adam_step(struct AdamOptimizer *optimizer, float ***weights, float ***grads, float *layer_sizes, int layer_sizes_rows, int layer_sizes_cols, float max_change) {
    const float b1 = optimizer->b1;
    const float b2 = optimizer->b2;
    const float lr = optimizer->lr;
    const float eps = optimizer->eps;
    const int epoch = ++optimizer->epoch;
    const float b1_pow = fast_pow(b1, epoch);
    const float b2_pow = fast_pow(b2, epoch);
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

        // Universal scalar processing
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n_inputs; i++) {
            float* __restrict weights_row = layer_weights[i];
            float* __restrict grads_row = layer_grads[i];
            float* __restrict m_row = layer_m[i];
            float* __restrict v_row = layer_v[i];

            #pragma omp simd
            for (int j = 0; j < n_neurons; j++) {
                const float grad = grads_row[j];
                const float grad_sq = grad * grad;

                // Calculation of moments
                const float new_m = b1 * m_row[j] + b1_minus_1 * grad;
                const float new_v = b2 * v_row[j] + b2_minus_1 * grad_sq;

                // Offset correction
                const float m_hat = new_m * inv_1mb1;
                const float v_hat = new_v * inv_1mb2;

                // Update of weights
                const float sqrt_v_hat = sqrtf(v_hat);
                const float delta = lr * m_hat / (sqrt_v_hat + eps);
                const float clipped_delta = fminf(fmaxf(delta, -max_change), max_change);

                // Saving results
                m_row[j] = new_m;
                v_row[j] = new_v;
                weights_row[j] -= clipped_delta;
            }
        }
    }
}
