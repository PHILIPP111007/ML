#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include "functions.h"

// Platform-dependent optimizations
#if defined(__x86_64__) || defined(__i386__)
    #define USE_x86
    #include <immintrin.h>
#elif defined(__aarch64__) || defined(__ARM_NEON)
    #define USE_ARM
    #include <arm_neon.h>
#endif


inline void matmul(float **restrict A, float **restrict B, float **restrict C, int rows_A, int cols_A, int rows_B, int cols_B) {
    // Checking compatibility of matrix sizes
    if (cols_A != rows_B) {
        fprintf(stderr, "Matrix dimensions mismatch: %d != %d\n", cols_A, rows_B);
        return;
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < rows_A; i++) {
        float *restrict a_row = A[i];
        float *restrict c_row = C[i];

        #ifdef USE_x86
            // AVX2 implementation for x86 (8 elements)
            for (int k = 0; k < cols_A; k++) {
                float *restrict b_row = B[k];
                __m256 a_val = _mm256_set1_ps(a_row[k]);

                int j = 0;
                for (; j <= cols_B - 8; j += 8) {
                    __m256 c = _mm256_loadu_ps(&c_row[j]);
                    __m256 b = _mm256_loadu_ps(&b_row[j]);
                    c = _mm256_fmadd_ps(a_val, b, c);
                    _mm256_storeu_ps(&c_row[j], c);
                }

                // Remaining elements
                #pragma omp simd
                for (int i = j; i < cols_B; i++) {
                    c_row[i] += a_row[k] * b_row[i];
                }
            }
        #elif defined(USE_ARM)
            // Implementation for ARM (4 elements)
            for (int k = 0; k < cols_A; k++) {
                float *restrict b_row = B[k];
                float32x4_t a_val = vdupq_n_f32(a_row[k]);

                int j = 0;
                for (; j <= cols_B - 4; j += 4) {
                    float32x4_t c = vld1q_f32(&c_row[j]);
                    float32x4_t b = vld1q_f32(&b_row[j]);
                    c = vmlaq_f32(c, a_val, b);
                    vst1q_f32(&c_row[j], c);
                }

                // Remaining elements
                #pragma omp simd
                for (int i = j; i < cols_B; i++) {
                    c_row[i] += a_row[k] * b_row[i];
                }
            }
        #else
            // Universal scalar implementation
            for (int k = 0; k < cols_A; k++) {
                float *b_row = B[k];
                float a_val = a_row[k];

                #pragma omp simd
                for (int j = 0; j < cols_B; j++) {
                    c_row[j] += a_val * b_row[j];
                }
            }
        #endif
    }
}

inline float sum(float **matrix, int rows, int cols) {
    float n = 0.0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            n += matrix[i][j];
        }
    }
    return n;
}

inline float *sum_axis_0(float **matrix, int rows, int cols) {
    float *result = malloc(cols * sizeof(float));

    for (int j = 0; j < cols; j++) {
        result[j] = 0;
        for (int i = 0; i < rows; i++) {
            result[j] += matrix[i][j];
        }
    }
    return result;
}

inline float mean(float *arr, int len) {
    if (len == 0) {
        return 0.0;
    }
    float sum = 0.0;
    for (int i = 0; i < len; i++) {
        sum += arr[i];
    }
    return sum / len;
}

inline int argmax(float *arr, int size) {
    if (size <= 0) {
        return -1;
    }

    int max_idx = 0;
    for (int i = 1; i < size; i++) {
        if (arr[i] > arr[max_idx]) {
            max_idx = i;
        }
    }
    return max_idx;
}

inline float safe_update(float number, float max_change) {
    return fmaxf(fminf(number, max_change), -max_change);
}

inline void apply_dropout(float **y, int matrix_rows, int n_neurons, float dropout) {
    for (int i = 0; i < matrix_rows; i++) {
        for (int j = 0; j < n_neurons; j++) {
            float random = (double)rand() / RAND_MAX;
            if (random > dropout) {
                y[i][j] = 0.0;
            }
        }
    }
}

inline float **create_matrix(int rows, int cols) {
    float **matrix = malloc(rows * sizeof(float*));
    float *data = malloc(rows * cols * sizeof(float));
    
    // Setting up line pointers
    for (int i = 0; i < rows; i++) {
        matrix[i] = &data[i * cols];
    }

    return matrix;
}

inline void free_matrix(float **matrix) {
    for (int i = 0; i < sizeof(matrix) / sizeof(matrix[0]); i++) {
        free(matrix[i]);
    }
    free(matrix);
}

inline float **transpose(float **original_matrix, int rows, int cols) {
    float **transposed_matrix = create_matrix(cols, rows);

    #pragma omp parallel for schedule(static)
    for (int j = 0; j < cols; j++) {
        #pragma omp simd
        for (int i = 0; i < rows; i++) {
            transposed_matrix[j][i] = original_matrix[i][j];
        }
    }
    return transposed_matrix;
}
