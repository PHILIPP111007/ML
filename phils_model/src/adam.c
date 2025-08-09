#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "functions.h"
#include "adam.h"

// Platform-dependent optimizations
#if defined(__x86_64__) || defined(__i386__)
    #define USE_x86
    #include <immintrin.h>
#elif defined(__aarch64__) || defined(__ARM_NEON)
    #define USE_ARM
    #include <arm_neon.h>
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

#ifdef USE_ARM
    inline void adam_step(
        struct AdamOptimizer *__restrict optimizer,
        float ***__restrict weights,
        float ***__restrict grads,
        float *__restrict layer_sizes,
        int layer_sizes_rows,
        int layer_sizes_cols,
        float max_change
    ) {
        // Get the optimizer parameters
        const float b1 = optimizer->b1;
        const float b2 = optimizer->b2;
        const float lr = optimizer->lr;
        const float eps = optimizer->eps;
        const int epoch = ++optimizer->epoch;

        // Pre-calculate frequently used values
        const float b1_pow = fast_pow(b1, epoch);
        const float b2_pow = fast_pow(b2, epoch);
        const float inv_1mb1 = 1.0f / (1.0f - b1_pow + 1e-10f);
        const float inv_1mb2 = 1.0f / (1.0f - b2_pow + 1e-10f);
        const float b1_minus_1 = 1.0f - b1;
        const float b2_minus_1 = 1.0f - b2;
        const float neg_max_change = -max_change;

        // Vectorized constants
        const float32x4_t vb1 = vdupq_n_f32(b1);
        const float32x4_t vb2 = vdupq_n_f32(b2);
        const float32x4_t vb1_minus_1 = vdupq_n_f32(b1_minus_1);
        const float32x4_t vb2_minus_1 = vdupq_n_f32(b2_minus_1);
        const float32x4_t vlr = vdupq_n_f32(lr);
        const float32x4_t veps = vdupq_n_f32(eps);
        const float32x4_t vmax_change = vdupq_n_f32(max_change);
        const float32x4_t vneg_max_change = vdupq_n_f32(neg_max_change);
        const float32x4_t vinv_1mb1 = vdupq_n_f32(inv_1mb1);
        const float32x4_t vinv_1mb2 = vdupq_n_f32(inv_1mb2);

        // Use cache-friendly chunk size (64 bytes / sizeof(float))
        const int chunk_size = 64 / sizeof(float);

        #pragma omp parallel for schedule(guided) if(layer_sizes_rows > 8)
        for (register int layer_index = 0; layer_index < layer_sizes_rows; layer_index++) {
            const int n_inputs = (int)layer_sizes[layer_index * layer_sizes_cols];
            const int n_neurons = (int)layer_sizes[layer_index * layer_sizes_cols + 1];

            float ** __restrict layer_weights = weights[layer_index];
            float ** __restrict layer_grads = grads[layer_index];
            float ** __restrict layer_m = optimizer->m[layer_index];
            float ** __restrict layer_v = optimizer->v[layer_index];

            #pragma omp parallel for schedule(guided) if(n_inputs > 8)
            for (int i = 0; i < n_inputs; i += chunk_size) {
                const int i_end = (i + chunk_size < n_inputs) ? i + chunk_size : n_inputs;

                #pragma omp parallel for schedule(guided) if(i_end > 8)
                for (int ii = i; ii < i_end; ii++) {
                    float *__restrict weights_row = layer_weights[ii];
                    float *__restrict grads_row = layer_grads[ii];
                    float *__restrict m_row = layer_m[ii];
                    float *__restrict v_row = layer_v[ii];

                    int jj = 0;
                    // Vectorized processing (4 elements at a time)
                    #pragma omp parallel for schedule(guided) if(n_neurons > 8)
                    for (int j = 0; j <= n_neurons - 4; j += 4) {
                        // Loading data
                        float32x4_t grad = vld1q_f32(grads_row + j);
                        float32x4_t m = vld1q_f32(m_row + j);
                        float32x4_t v = vld1q_f32(v_row + j);
                        float32x4_t w = vld1q_f32(weights_row + j);

                        // Calculate the square of the gradient
                        float32x4_t grad_sq = vmulq_f32(grad, grad);

                        // Updating the moments
                        m = vmlaq_f32(vmulq_f32(vb1, m), vb1_minus_1, grad);
                        v = vmlaq_f32(vmulq_f32(vb2, v), vb2_minus_1, grad_sq);

                        // Offset correction
                        float32x4_t m_hat = vmulq_f32(m, vinv_1mb1);
                        float32x4_t v_hat = vmulq_f32(v, vinv_1mb2);

                        // Calculating the update
                        float32x4_t sqrt_v_hat = vsqrtq_f32(v_hat);
                        float32x4_t delta = vdivq_f32(
                            vmulq_f32(vlr, m_hat),
                            vaddq_f32(sqrt_v_hat, veps)
                        );

                        // Limiting the change in weights
                        delta = vminq_f32(vmaxq_f32(delta, vneg_max_change), vmax_change);

                        // Update weights and save moments
                        w = vsubq_f32(w, delta);
                        vst1q_f32(weights_row + j, w);
                        vst1q_f32(m_row + j, m);
                        vst1q_f32(v_row + j, v);

                        jj = j;
                    }

                    // Processing remaining elements
                    #pragma omp simd
                    for (int j = jj; j < n_neurons; j++) {
                        const float grad = grads_row[j];
                        const float grad_sq = grad * grad;

                        // Updating the moments
                        const float new_m = b1 * m_row[j] + b1_minus_1 * grad;
                        const float new_v = b2 * v_row[j] + b2_minus_1 * grad_sq;

                        // Offset correction
                        const float m_hat = new_m * inv_1mb1;
                        const float v_hat = new_v * inv_1mb2;

                        // Calculating the update
                        const float sqrt_v_hat = sqrtf(v_hat);
                        float delta = lr * m_hat / (sqrt_v_hat + eps);

                        // Limiting the change in weights
                        delta = delta > max_change ? max_change : delta;
                        delta = delta < neg_max_change ? neg_max_change : delta;

                        // Save the results
                        m_row[j] = new_m;
                        v_row[j] = new_v;
                        weights_row[j] -= delta;
                    }
                }
            }
        }
    }
#elif defined(USE_x86)
    inline void adam_step(
        struct AdamOptimizer *__restrict optimizer,
        float ***__restrict weights,
        float ***__restrict grads,
        float *__restrict layer_sizes,
        int layer_sizes_rows,
        int layer_sizes_cols,
        float max_change
    ) {
        // Get the optimizer parameters
        const float b1 = optimizer->b1;
        const float b2 = optimizer->b2;
        const float lr = optimizer->lr;
        const float eps = optimizer->eps;
        const int epoch = ++optimizer->epoch;

        // Pre-calculate frequently used values
        const float b1_pow = fast_pow(b1, epoch);
        const float b2_pow = fast_pow(b2, epoch);
        const float inv_1mb1 = 1.0f / (1.0f - b1_pow + 1e-10f);
        const float inv_1mb2 = 1.0f / (1.0f - b2_pow + 1e-10f);
        const float b1_minus_1 = 1.0f - b1;
        const float b2_minus_1 = 1.0f - b2;
        const float neg_max_change = -max_change;

        // Vectorized constants
        const __m256 vb1 = _mm256_set1_ps(b1);
        const __m256 vb2 = _mm256_set1_ps(b2);
        const __m256 vb1_minus_1 = _mm256_set1_ps(b1_minus_1);
        const __m256 vb2_minus_1 = _mm256_set1_ps(b2_minus_1);
        const __m256 vlr = _mm256_set1_ps(lr);
        const __m256 veps = _mm256_set1_ps(eps);
        const __m256 vmax_change = _mm256_set1_ps(max_change);
        const __m256 vneg_max_change = _mm256_set1_ps(neg_max_change);
        const __m256 vinv_1mb1 = _mm256_set1_ps(inv_1mb1);
        const __m256 vinv_1mb2 = _mm256_set1_ps(inv_1mb2);

        // Use cache-friendly chunk size (64 bytes / sizeof(float))
        const int chunk_size = 64 / sizeof(float);

        #pragma omp parallel for schedule(guided) if(layer_sizes_rows > 4)
        for (int layer_index = 0; layer_index < layer_sizes_rows; layer_index++) {
            const int n_inputs = (int)layer_sizes[layer_index * layer_sizes_cols];
            const int n_neurons = (int)layer_sizes[layer_index * layer_sizes_cols + 1];

            float ** __restrict layer_weights = weights[layer_index];
            float ** __restrict layer_grads = grads[layer_index];
            float ** __restrict layer_m = optimizer->m[layer_index];
            float ** __restrict layer_v = optimizer->v[layer_index];

            #pragma omp parallel for schedule(guided) if(n_inputs > 8)
            for (int i = 0; i < n_inputs; i += chunk_size) {
                const int i_end = (i + chunk_size < n_inputs) ? i + chunk_size : n_inputs;

                #pragma omp parallel for schedule(guided) if(i_end > 8)
                for (int ii = i; ii < i_end; ii++) {
                    float *__restrict weights_row = layer_weights[ii];
                    float *__restrict grads_row = layer_grads[ii];
                    float *__restrict m_row = layer_m[ii];
                    float *__restrict v_row = layer_v[ii];

                    int jj = 0;
                    // Vectorized processing (8 elements at a time)
                    #pragma omp parallel for schedule(guided) if(n_neurons > 8)
                    for (int j = 0; j <= n_neurons - 8; j += 8) {
                        // Loading data
                        __m256 grad = _mm256_loadu_ps(grads_row + j);
                        __m256 m = _mm256_loadu_ps(m_row + j);
                        __m256 v = _mm256_loadu_ps(v_row + j);
                        __m256 w = _mm256_loadu_ps(weights_row + j);

                        // Calculate the square of the gradient
                        __m256 grad_sq = _mm256_mul_ps(grad, grad);

                        // Updating Moments (FMA Instructions)
                        m = _mm256_fmadd_ps(vb1_minus_1, grad, _mm256_mul_ps(vb1, m));
                        v = _mm256_fmadd_ps(vb2_minus_1, grad_sq, _mm256_mul_ps(vb2, v));

                        // Offset correction
                        __m256 m_hat = _mm256_mul_ps(m, vinv_1mb1);
                        __m256 v_hat = _mm256_mul_ps(v, vinv_1mb2);

                        // Calculating the update
                        __m256 sqrt_v_hat = _mm256_sqrt_ps(v_hat);
                        __m256 delta = _mm256_div_ps(
                            _mm256_mul_ps(vlr, m_hat),
                            _mm256_add_ps(sqrt_v_hat, veps)
                        );

                        // Limiting the change in weights
                        delta = _mm256_min_ps(_mm256_max_ps(delta, vneg_max_change), vmax_change);

                        // Update weights and save moments
                        w = _mm256_sub_ps(w, delta);
                        _mm256_storeu_ps(weights_row + j, w);
                        _mm256_storeu_ps(m_row + j, m);
                        _mm256_storeu_ps(v_row + j, v);

                        jj = j;
                    }

                    // Processing the remaining elements (4 at a time)
                    for (int j = jj; j <= n_neurons - 4; j += 4) {
                        __m128 grad = _mm_loadu_ps(grads_row + j);
                        __m128 m = _mm_loadu_ps(m_row + j);
                        __m128 v = _mm_loadu_ps(v_row + j);
                        __m128 w = _mm_loadu_ps(weights_row + j);

                        __m128 grad_sq = _mm_mul_ps(grad, grad);
                        
                        m = _mm_fmadd_ps(_mm_set1_ps(b1_minus_1), grad, _mm_mul_ps(_mm_set1_ps(b1), m));
                        v = _mm_fmadd_ps(_mm_set1_ps(b2_minus_1), grad_sq, _mm_mul_ps(_mm_set1_ps(b2), v));
                        
                        __m128 m_hat = _mm_mul_ps(m, _mm_set1_ps(inv_1mb1));
                        __m128 v_hat = _mm_mul_ps(v, _mm_set1_ps(inv_1mb2));
                        
                        __m128 sqrt_v_hat = _mm_sqrt_ps(v_hat);
                        __m128 delta = _mm_div_ps(
                            _mm_mul_ps(_mm_set1_ps(lr), m_hat),
                            _mm_add_ps(sqrt_v_hat, _mm_set1_ps(eps))
                        );
                        
                        delta = _mm_min_ps(_mm_max_ps(delta, _mm_set1_ps(neg_max_change)), _mm_set1_ps(max_change));
                        
                        w = _mm_sub_ps(w, delta);
                        _mm_storeu_ps(weights_row + j, w);
                        _mm_storeu_ps(m_row + j, m);
                        _mm_storeu_ps(v_row + j, v);

                        jj = j;
                    }

                    // Processing last items (1 at a time)
                    #pragma omp simd
                    for (int j = jj; j < n_neurons; j++) {
                        const float grad = grads_row[j];
                        const float grad_sq = grad * grad;

                        // Updating the moments
                        const float new_m = b1 * m_row[j] + b1_minus_1 * grad;
                        const float new_v = b2 * v_row[j] + b2_minus_1 * grad_sq;

                        // Offset correction
                        const float m_hat = new_m * inv_1mb1;
                        const float v_hat = new_v * inv_1mb2;

                        // Calculating the update
                        const float sqrt_v_hat = sqrtf(v_hat);
                        float delta = lr * m_hat / (sqrt_v_hat + eps);

                        // Limiting the change in weights
                        delta = delta > max_change ? max_change : delta;
                        delta = delta < neg_max_change ? neg_max_change : delta;

                        // Save the results
                        m_row[j] = new_m;
                        v_row[j] = new_v;
                        weights_row[j] -= delta;
                    }
                }
            }
        }
    }

#else
    inline void adam_step(
        struct AdamOptimizer *__restrict optimizer,
        float ***__restrict weights,
        float ***__restrict grads,
        float *__restrict layer_sizes,
        int layer_sizes_rows,
        int layer_sizes_cols,
        float max_change
    ) {
        // Cache optimizer parameters
        const register float b1 = optimizer->b1;
        const register float b2 = optimizer->b2;
        const register float lr = optimizer->lr;
        const register float eps = optimizer->eps;
        const register int epoch = ++optimizer->epoch;

        // Precompute bias corrections
        const register float b1_pow = fast_pow(b1, epoch);
        const register float b2_pow = fast_pow(b2, epoch);
        const register float inv_1mb1 = 1.0f / (1.0f - b1_pow + 1e-10f);
        const register float inv_1mb2 = 1.0f / (1.0f - b2_pow + 1e-10f);
        const register float b1_minus_1 = 1.0f - b1;
        const register float b2_minus_1 = 1.0f - b2;

        // Use cache-friendly chunk size (64 bytes / sizeof(float))
        const register int chunk_size = 64 / sizeof(float);

        // Process layers with guided scheduling for load balancing
        #pragma omp parallel for schedule(guided) if(layer_sizes_rows > 8)
        for (register int layer_index = 0; layer_index < layer_sizes_rows; layer_index++) {
            const register int n_inputs = (int)layer_sizes[layer_index * layer_sizes_cols];
            const register int n_neurons = (int)layer_sizes[layer_index * layer_sizes_cols + 1];

            float ** __restrict layer_weights = weights[layer_index];
            float ** __restrict layer_grads = grads[layer_index];
            float ** __restrict layer_m = optimizer->m[layer_index];
            float ** __restrict layer_v = optimizer->v[layer_index];

            // Process rows in chunks
            #pragma omp parallel for schedule(guided) if(n_inputs > 8)
            for (register int i = 0; i < n_inputs; i += chunk_size) {
                const register int i_end = (i + chunk_size < n_inputs) ? i + chunk_size : n_inputs;

                #pragma omp parallel for schedule(guided) if(i_end > 8)
                for (register int ii = i; ii < i_end; ii++) {
                    float *__restrict weights_row = layer_weights[ii];
                    float *__restrict grads_row = layer_grads[ii];
                    float *__restrict m_row = layer_m[ii];
                    float *__restrict v_row = layer_v[ii];

                    #pragma omp parallel for schedule(guided) if(n_neurons > 8)
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
#endif

inline void adam_step_gpu(
    struct AdamOptimizer *__restrict optimizer,
    float ***__restrict weights,
    float ***__restrict grads,
    float *__restrict layer_sizes,
    int layer_sizes_rows,
    int layer_sizes_cols,
    float max_change,
    cl_context context,
    cl_command_queue queue,
    cl_program program
) {
    // Cache optimizer parameters
    const float b1 = optimizer->b1;
    const float b2 = optimizer->b2;
    const float lr = optimizer->lr;
    const float eps = optimizer->eps;
    const int epoch = ++optimizer->epoch;

    // Precompute bias corrections
    const float b1_pow = fast_pow(b1, epoch);
    const float b2_pow = fast_pow(b2, epoch);
    const float inv_1mb1 = 1.0f / (1.0f - b1_pow + 1e-10f);
    const float inv_1mb2 = 1.0f / (1.0f - b2_pow + 1e-10f);
    const float b1_minus_1 = 1.0f - b1;
    const float b2_minus_1 = 1.0f - b2;


    for (int layer_index = 0; layer_index < layer_sizes_rows; layer_index++) {
        
        const int n_inputs = (int)layer_sizes[layer_index * layer_sizes_cols];
        const int n_neurons = (int)layer_sizes[layer_index * layer_sizes_cols + 1];

        float **layer_weights = weights[layer_index];
        float **layer_grads = grads[layer_index];
        float **layer_m = optimizer->m[layer_index];
        float **layer_v = optimizer->v[layer_index];

        // We determine the total volume of data
        const int total_elements = n_inputs * n_neurons;

        float *layer_weights_vec = malloc(total_elements * sizeof(float));
        float *layer_grads_vec = malloc(total_elements * sizeof(float));
        float *layer_m_vec = malloc(total_elements * sizeof(float));
        float *layer_v_vec = malloc(total_elements * sizeof(float));

        for (int i = 0; i < n_inputs; i++) {
            for (int j = 0; j < n_neurons; j++) {
                layer_weights_vec[i * n_neurons + j] = layer_weights[i][j];
                layer_grads_vec[i * n_neurons + j] = layer_grads[i][j];
                layer_m_vec[i * n_neurons + j] = layer_m[i][j];
                layer_v_vec[i * n_neurons + j] = layer_v[i][j];
            }
        }

        // Get the kernel
        cl_kernel kernel = clCreateKernel(program, "adam_step_gpu", NULL);

        // Preparing OpenCL buffers
        cl_mem weights_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, total_elements * sizeof(float), layer_weights_vec, NULL);
        cl_mem grads_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, total_elements * sizeof(float), layer_grads_vec, NULL);
        cl_mem m_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, total_elements * sizeof(float), layer_m_vec, NULL);
        cl_mem v_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, total_elements * sizeof(float), layer_v_vec, NULL);

        // Setting kernel arguments
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &weights_buf);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &grads_buf);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &m_buf);
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &v_buf);
        clSetKernelArg(kernel, 4, sizeof(float), &b1);
        clSetKernelArg(kernel, 5, sizeof(float), &b2);
        clSetKernelArg(kernel, 6, sizeof(float), &lr);
        clSetKernelArg(kernel, 7, sizeof(float), &eps);
        clSetKernelArg(kernel, 8, sizeof(float), &inv_1mb1);
        clSetKernelArg(kernel, 9, sizeof(float), &inv_1mb2);
        clSetKernelArg(kernel, 10, sizeof(float), &b1_minus_1);
        clSetKernelArg(kernel, 11, sizeof(float), &b2_minus_1);
        clSetKernelArg(kernel, 12, sizeof(float), &max_change);
        clSetKernelArg(kernel, 13, sizeof(int), &total_elements);

        // Working Grid Settings
        size_t global_work_size[] = { total_elements };

        // Launching the OpenCL kernel
        clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);

        // Reading the results back
        clEnqueueReadBuffer(queue, weights_buf, CL_TRUE, 0, total_elements * sizeof(float), layer_weights_vec, 0, NULL, NULL);
        clEnqueueReadBuffer(queue, grads_buf, CL_TRUE, 0, total_elements * sizeof(float), layer_grads_vec, 0, NULL, NULL);
        clEnqueueReadBuffer(queue, m_buf, CL_TRUE, 0, total_elements * sizeof(float), layer_m_vec, 0, NULL, NULL);
        clEnqueueReadBuffer(queue, v_buf, CL_TRUE, 0, total_elements * sizeof(float), layer_v_vec, 0, NULL, NULL);

        clReleaseMemObject(weights_buf);
        clReleaseMemObject(grads_buf);
        clReleaseMemObject(m_buf);
        clReleaseMemObject(v_buf);

        for (int i = 0; i < n_inputs; i++) {
            for (int j = 0; j < n_neurons; j++) {
                layer_weights[i][j] = layer_weights_vec[i * n_neurons + j];
                layer_grads[i][j] = layer_grads_vec[i * n_neurons + j];
                layer_m[i][j] = layer_m_vec[i * n_neurons + j];
                layer_v[i][j] = layer_v_vec[i * n_neurons + j];
            }
        }

        free(layer_weights_vec);
        free(layer_grads_vec);
        free(layer_m_vec);
        free(layer_v_vec);

        clReleaseKernel(kernel);
    }
}
