#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include "logger.h"
#include "functions.h"

// Platform-dependent optimizations
#if defined(__x86_64__) || defined(__i386__)
    #define USE_x86
    #include <immintrin.h>
#elif defined(__aarch64__) || defined(__ARM_NEON)
    #define USE_ARM
    #include <arm_neon.h>
#endif

#define LOCAL_BLOCK_SIZE 8


inline void check_if_null(float *pointer, char *pointer_name) {
    if (pointer == NULL) {
        char *s = (char*)malloc(100 * sizeof(char));
        sprintf(s, "Failed to create pointer %s.\n", pointer_name);
        logger_error(s);
        exit(EXIT_FAILURE);
    }
}

inline int get_array_size(float *array) {
    return sizeof(array) / sizeof(array[0]);
}

inline void check_if_index_out_of_bounds(int index, int array_size, char *array_name) {
    if (index >= array_size) {
        char *s = (char*)malloc(100 * sizeof(char));
        sprintf(s, "Index out of bounds for array %s.\n", array_name);
        logger_error(s);
    }
}

inline void matmul(float **__restrict A, float **__restrict B, float **__restrict C, int rows_A, int cols_A, int rows_B, int cols_B) {
    // Checking compatibility of matrix sizes
    if (cols_A != rows_B) {
        fprintf(stderr, "Matrix dimensions mismatch: %d != %d\n", cols_A, rows_B);
        return;
    }

    for (register int i = 0; i < rows_A; i++) {
        float *__restrict a_row = A[i];
        float *__restrict c_row = C[i];

        #ifdef USE_x86
            // AVX2 implementation for x86 (8 elements)
            for (register int k = 0; k < cols_A; k++) {
                float *__restrict b_row = B[k];
                __m256 a_val = _mm256_set1_ps(a_row[k]);

                register int j = 0;
                for (; j <= cols_B - 8; j += 8) {
                    __m256 c = _mm256_loadu_ps(&c_row[j]);
                    __m256 b = _mm256_loadu_ps(&b_row[j]);
                    c = _mm256_fmadd_ps(a_val, b, c);
                    _mm256_storeu_ps(&c_row[j], c);
                }

                // Remaining elements
                #pragma omp simd
                for (register int i = j; i < cols_B; i++) {
                    c_row[i] += a_row[k] * b_row[i];
                }
            }
        #elif defined(USE_ARM)
            // Implementation for ARM (4 elements)
            for (register int k = 0; k < cols_A; k++) {
                float *__restrict b_row = B[k];
                register float32x4_t a_val = vdupq_n_f32(a_row[k]);

                register int j = 0;
                for (; j <= cols_B - 4; j += 4) {
                    register float32x4_t c = vld1q_f32(&c_row[j]);
                    register float32x4_t b = vld1q_f32(&b_row[j]);
                    c = vmlaq_f32(c, a_val, b);
                    vst1q_f32(&c_row[j], c);
                }

                // Remaining elements
                #pragma omp simd
                for (register int i = j; i < cols_B; i++) {
                    c_row[i] += a_row[k] * b_row[i];
                }
            }
        #else
            // Universal scalar implementation
            for (register int k = 0; k < cols_A; k++) {
                float *b_row = B[k];
                register float a_val = a_row[k];

                #pragma omp simd
                for (register int j = 0; j < cols_B; j++) {
                    register b_val = b_row[j];
                    c_row[j] += a_val * b_val;
                }
            }
        #endif
    }
}

// Matrix Multiplication Using OpenCL
inline void matmul_gpu(cl_context context, cl_command_queue queue, cl_program program, float *A, cl_mem d_B, float *C, int ROWS_A, int COLS_A, int ROWS_B, int COLS_B, int layer_index) {
    // Get the kernel
    cl_kernel kernel = clCreateKernel(program, "matmul_gpu", NULL);

    // Creating buffers for storing matrices
    cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ROWS_A * COLS_A * sizeof(float), A, NULL);
    cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, ROWS_A * COLS_B * sizeof(float), NULL, NULL);

    // Setting kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);
    clSetKernelArg(kernel, 3, sizeof(int), &ROWS_A);
    clSetKernelArg(kernel, 4, sizeof(int), &COLS_A);
    clSetKernelArg(kernel, 5, sizeof(int), &COLS_B);
    clSetKernelArg(kernel, 6, sizeof(int), &layer_index);

    // Step 9: Determine the size of the thread grid
    size_t global_size[] = {ROWS_A, COLS_B}; // Working flow volume
    size_t local_size[] = {LOCAL_BLOCK_SIZE, LOCAL_BLOCK_SIZE}; // Working flow volume

    // Launch the kernel
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, local_size, 0, NULL, NULL);

    // Reading the result
    clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, ROWS_A * COLS_B * sizeof(float), C, 0, NULL, NULL);

    // Resource cleanup
    clReleaseMemObject(d_A);
    clReleaseMemObject(d_C);

    clReleaseKernel(kernel);
}

inline float sum(float **__restrict matrix, int rows, int cols) {
    register float n = 0.0f;
    for (register int i = 0; i < rows; i++) {
        #pragma omp simd
        for (register int j = 0; j < cols; j++) {
            n += matrix[i][j];
        }
    }
    return n;
}

inline float *sum_axis_0(float **__restrict matrix, int rows, int cols) {
    float *result = malloc(cols * sizeof(float));
    check_if_null((float *)result, "result");

    for (register int j = 0; j < cols; j++) {
        result[j] = 0.0f;
        #pragma omp simd
        for (register int i = 0; i < rows; i++) {
            result[j] += matrix[i][j];
        }
    }
    return result;
}

inline float mean(float *__restrict arr, int len) {
    if (len == 0) {
        return 0.0f;
    }
    register float sum = 0.0f;

    #pragma omp simd
    for (register int i = 0; i < len; i++) {
        sum += arr[i];
    }
    return sum / len;
}

inline int argmax(float *__restrict arr, int size) {
    if (size <= 0) {
        return -1;
    }

    int max_idx = 0;
    for (register int i = 1; i < size; i++) {
        if (arr[i] > arr[max_idx]) {
            max_idx = i;
        }
    }
    return max_idx;
}

inline float safe_update(float number, float max_change) {
    return fmax(fmin(number, max_change), -max_change);
}

inline void apply_dropout(float **__restrict y, int matrix_rows, int n_neurons, float dropout) {
    for (register int i = 0; i < matrix_rows; i++) {
        for (register int j = 0; j < n_neurons; j++) {
            float random = (double)rand() / RAND_MAX;
            if (random > dropout) {
                y[i][j] = 0.0f;
            }
        }
    }
}

inline float **create_matrix(int rows, int cols) {
    float **__restrict matrix = malloc(rows * sizeof(float*));
    check_if_null((float *)matrix, "matrix");

    float *__restrict data = malloc(rows * cols * sizeof(float));
    check_if_null((float *)data, "data");

    // Setting up line pointers
    #pragma omp simd
    for (register int i = 0; i < rows; i++) {
        matrix[i] = &data[i * cols];
    }

    return matrix;
}

inline void free_matrix(float **__restrict matrix) {
    for (register int i = 0; i < sizeof(matrix) / sizeof(matrix[0]); i++) {
        free(matrix[i]);
    }
    free(matrix);
}

inline float **transpose(float **__restrict original_matrix, int rows, int cols) {
    float **__restrict transposed_matrix = create_matrix(cols, rows);

    for (register int j = 0; j < cols; j++) {
        #pragma omp simd
        for (register int i = 0; i < rows; i++) {
            transposed_matrix[j][i] = original_matrix[i][j];
        }
    }
    return transposed_matrix;
}

inline float *get_weights_vec(float ***weights, int layer_sizes_rows, int layer_sizes_cols, float *layer_sizes) {
    int total_elements_per_sample = 0;
    for (int layer_index = 0; layer_index < layer_sizes_rows; layer_index++) {
        const int n_inputs = (int)layer_sizes[layer_index * layer_sizes_cols];
        const int n_neurons = (int)layer_sizes[layer_index * layer_sizes_cols + 1];
        total_elements_per_sample += n_inputs * n_neurons;
    }

    int total_elements_weights = total_elements_per_sample;
    float *weights_vec = malloc(total_elements_weights * sizeof(float));
    check_if_null((float *)weights_vec, "weights_vec");

    int current_weight_offset = 0;
    for (int layer_index = 0; layer_index < layer_sizes_rows; layer_index++) {
        const int n_inputs = (int)layer_sizes[layer_index * layer_sizes_cols];
        const int n_neurons = (int)layer_sizes[layer_index * layer_sizes_cols + 1];

        for (int i = 0; i < n_inputs; i++) {
            #pragma omp simd
            for (int j = 0; j < n_neurons; j++) {
                int index = current_weight_offset + i * n_neurons + j;
                weights_vec[index] = weights[layer_index][i][j];
            }
        }
        current_weight_offset += n_inputs * n_neurons;
    }

    return weights_vec;
}

inline float *get_weights_transposed_vec(float ***weights_transposed, int layer_sizes_rows, int layer_sizes_cols, float *layer_sizes) {
    int total_elements_per_sample = 0;
    for (int layer_index = 0; layer_index < layer_sizes_rows; layer_index++) {
        const int n_inputs = (int)layer_sizes[layer_index * layer_sizes_cols];
        const int n_neurons = (int)layer_sizes[layer_index * layer_sizes_cols + 1];
        total_elements_per_sample += n_inputs * n_neurons;
    }

    int total_elements_weights = total_elements_per_sample;
    float *weights_transposed_vec = malloc(total_elements_weights * sizeof(float));
    check_if_null((float *)weights_transposed_vec, "weights_transposed_vec");

    int current_weight_offset = 0;
    for (int layer_index = 0; layer_index < layer_sizes_rows; layer_index++) {
        const int n_inputs = (int)layer_sizes[layer_index * layer_sizes_cols];
        const int n_neurons = (int)layer_sizes[layer_index * layer_sizes_cols + 1];

        for (int i = 0; i < n_neurons; i++) {
            #pragma omp simd
            for (int j = 0; j < n_inputs; j++) {
                int index = current_weight_offset + i * n_inputs + j;
                weights_transposed_vec[index] = weights_transposed[layer_index][i][j];
            }
        }
        current_weight_offset += n_inputs * n_neurons;
    }

    return weights_transposed_vec;
}

inline cl_mem get_weights_vec_buf(float *weights_vec, int layer_sizes_rows, int layer_sizes_cols, float *layer_sizes, cl_context context) {
    int total_elements_per_sample = 0;
    for (int layer_index = 0; layer_index < layer_sizes_rows; layer_index++) {
        const int n_inputs = (int)layer_sizes[layer_index * layer_sizes_cols];
        const int n_neurons = (int)layer_sizes[layer_index * layer_sizes_cols + 1];
        total_elements_per_sample += n_inputs * n_neurons;
    }

    int total_elements_weights = total_elements_per_sample;

    cl_int err;
    cl_mem weights_vec_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, total_elements_weights * sizeof(float), weights_vec, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create weights buffer (Error: %d)\n", err);
        exit(EXIT_FAILURE);
    }

    return weights_vec_buf;
}

inline char *get_file_content(char *file_path) {
    FILE *fp = fopen(file_path, "rb");
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    rewind(fp);
    char *source = (char*)malloc(file_size + 1);
    fread(source, 1, file_size, fp);
    fclose(fp);

    return source;
}

inline float check_if_isnan(float number) {
    return isnan(number) ? 0.0f : number;
}
