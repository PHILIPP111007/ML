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

inline void adam_step(struct AdamOptimizer *__restrict optimizer, float ***__restrict weights, float ***__restrict grads, float *__restrict layer_sizes, int layer_sizes_rows, int layer_sizes_cols, float max_change) {
    // Cache optimizer parameters in registers
    const register float b1 = optimizer->b1;
    const register float b2 = optimizer->b2;
    const register float lr = optimizer->lr;
    const register float eps = optimizer->eps;
    const register int epoch = ++optimizer->epoch;

    // Precompute bias corrections with SIMD-friendly constants
    const register float b1_pow = fast_pow(b1, epoch);
    const register float b2_pow = fast_pow(b2, epoch);
    const register float inv_1mb1 = 1.0f / (1.0f - b1_pow + 1e-10f); // Add small epsilon to prevent division by zero
    const register float inv_1mb2 = 1.0f / (1.0f - b2_pow + 1e-10f);
    const register float b1_minus_1 = 1.0f - b1;
    const register float b2_minus_1 = 1.0f - b2;

    // Process layers
    for (register int layer_index = 0; layer_index < layer_sizes_rows; layer_index++) {
        const register int n_inputs = (int)layer_sizes[layer_index * layer_sizes_cols];
        const register int n_neurons = (int)layer_sizes[layer_index * layer_sizes_cols + 1];

        float ** __restrict layer_weights = weights[layer_index];
        float ** __restrict layer_grads = grads[layer_index];
        float ** __restrict layer_m = optimizer->m[layer_index];
        float ** __restrict layer_v = optimizer->v[layer_index];

        // Process rows in larger chunks for better cache utilization
        const register int chunk_size = 64 / sizeof(float);

        #pragma omp parallel for schedule(guided)
        for (register int i = 0; i < n_inputs; i += chunk_size) {
            const register int i_end = (i + chunk_size < n_inputs) ? i + chunk_size : n_inputs;

            // Inner loop over chunked rows
            for (register int ii = i; ii < i_end; ii++) {
                float *__restrict weights_row = layer_weights[ii];
                float *__restrict grads_row = layer_grads[ii];
                float *__restrict m_row = layer_m[ii];
                float *__restrict v_row = layer_v[ii];

                #pragma omp parallel for schedule(guided)
                for (register int j = 0; j < n_neurons; j += 4) {
                    if (j + 3 < n_neurons) {
                        #pragma omp simd aligned(weights_row, grads_row, m_row, v_row:64)
                        for (register int k = 0; k < 4; k++) {
                            const register int idx = j + k;
                            const register float grad = grads_row[idx];
                            const register float grad_sq = grad * grad;

                            m_row[idx] = b1 * m_row[idx] + b1_minus_1 * grad;
                            v_row[idx] = b2 * v_row[idx] + b2_minus_1 * grad_sq;

                            const register float m_hat = m_row[idx] * inv_1mb1;
                            const register float v_hat = v_row[idx] * inv_1mb2;
                            const register float delta = lr * m_hat / (sqrtf(v_hat) + eps);
                            weights_row[idx] -= fminf(fmaxf(delta, -max_change), max_change);
                        }
                    } else {
                        // Handle remaining elements
                        #pragma omp simd aligned(weights_row, grads_row, m_row, v_row:64)
                        for (register int k = j; k < n_neurons; k++) {
                            const register float grad = grads_row[k];
                            const register float grad_sq = grad * grad;

                            m_row[k] = b1 * m_row[k] + b1_minus_1 * grad;
                            v_row[k] = b2 * v_row[k] + b2_minus_1 * grad_sq;

                            const register float m_hat = m_row[k] * inv_1mb1;
                            const register float v_hat = v_row[k] * inv_1mb2;
                            const register float delta = lr * m_hat / (sqrtf(v_hat) + eps);
                            weights_row[k] -= fminf(fmaxf(delta, -max_change), max_change);
                        }
                    }
                }
            }
        }
    }
}