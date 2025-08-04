#include <stdlib.h>
#include <math.h>
#include "functions.h"
#include "adam.h"


#if defined(__x86_64__) || defined(__i386__)
    #define USE_x86
    #include <immintrin.h>


    inline float fast_sqrt(float x) {
        return _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(x)));
    }
#elif defined(__aarch64__) || defined(__ARM_NEON)
    #define USE_ARM
    #include <arm_neon.h>


    inline float fast_sqrt(float x) {
        return vgetq_lane_f32(vsqrtq_f32(vdupq_n_f32(x)), 0);
    }
#else
    // Universal implementation
    inline float fast_sqrt(float x) {
        return sqrtf(x);
    }
#endif

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

        // Vectorized Processing for ARM NEON
        #ifdef USE_ARM
            int jj = 0;
            int ii = 0;

            #pragma omp parallel for collapse(2)
            for (int i = 0; i < n_inputs; i++) {
                for (int j = 0; j <= n_neurons - 4; j += 4) {
                    float32x4_t grad = vld1q_f32(&layer_grads[i][j]);
                    float32x4_t m = vld1q_f32(&layer_m[i][j]);
                    float32x4_t v = vld1q_f32(&layer_v[i][j]);
                    float32x4_t w = vld1q_f32(&layer_weights[i][j]);

                    // Calculation of moments
                    float32x4_t grad_sq = vmulq_f32(grad, grad);
                    m = vmlaq_f32(vmulq_f32(vdupq_n_f32(b1), m), vdupq_n_f32(b1_minus_1), grad);
                    v = vmlaq_f32(vmulq_f32(vdupq_n_f32(b2), v), vdupq_n_f32(b2_minus_1), grad_sq);

                    // Offset correction
                    float32x4_t m_hat = vmulq_f32(m, vdupq_n_f32(inv_1mb1));
                    float32x4_t v_hat = vmulq_f32(v, vdupq_n_f32(inv_1mb2));

                    // Update of weights
                    float32x4_t sqrt_v = vsqrtq_f32(vaddq_f32(v_hat, vdupq_n_f32(eps)));
                    float32x4_t delta = vmulq_f32(vdupq_n_f32(lr), vdivq_f32(m_hat, sqrt_v));

                    // Restriction of change
                    float32x4_t clipped = vminq_f32(vmaxq_f32(delta, vdupq_n_f32(-max_change)), vdupq_n_f32(max_change));

                    // Saving results
                    w = vsubq_f32(w, clipped);
                    vst1q_f32(&layer_weights[i][j], w);
                    vst1q_f32(&layer_m[i][j], m);
                    vst1q_f32(&layer_v[i][j], v);

                    jj = j;
                    ii = i;
                }
            }

            // Processing remaining elements
            for (int i = ii; i < n_inputs; i++) {
                for (int j = jj; j < n_neurons; j++) {
                    const float grad = layer_grads[i][j];
                    const float grad_sq = grad * grad;

                    layer_m[i][j] = b1 * layer_m[i][j] + b1_minus_1 * grad;
                    layer_v[i][j] = b2 * layer_v[i][j] + b2_minus_1 * grad_sq;

                    const float m_hat = layer_m[i][j] * inv_1mb1;
                    const float v_hat = layer_v[i][j] * inv_1mb2;

                    const float delta = lr * m_hat / (fast_sqrt(v_hat) + eps);
                    layer_weights[i][j] -= fminf(fmaxf(delta, -max_change), max_change);
                }
            }
        #else
            // Universal scalar processing
            #pragma omp for collapse(2)
            for (int i = 0; i < n_inputs; i++) {
                for (int j = 0; j < n_neurons; j++) {
                    const float grad = layer_grads[i][j];
                    const float grad_sq = grad * grad;

                    layer_m[i][j] = b1 * layer_m[i][j] + b1_minus_1 * grad;
                    layer_v[i][j] = b2 * layer_v[i][j] + b2_minus_1 * grad_sq;

                    const float m_hat = layer_m[i][j] * inv_1mb1;
                    const float v_hat = layer_v[i][j] * inv_1mb2;

                    const float delta = lr * m_hat / (fast_sqrt(v_hat) + eps);
                    layer_weights[i][j] -= fminf(fmaxf(delta, -max_change), max_change);
                }
            }
        #endif
    }
}
