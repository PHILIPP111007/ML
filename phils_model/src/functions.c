#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include "functions.h"


inline void matmul(float **A, float **B, float **C, int rows_A, int cols_A, int rows_B, int cols_B) {
    if (cols_A != rows_B) {
        fprintf(stderr, "Matrix dimensions mismatch: %d != %d\n", cols_A, rows_B);
        return;
    }

    for (register int i = 0; i < rows_A; i++) {
        float *a = A[i];
        
        for (register int k = 0; k < cols_A; k++) {
            float *b = B[k];

            #pragma omp simd
            for (register int j = 0; j < cols_B; j++) {
                C[i][j] += a[k] * b[j];
            }
        }
    }
}

inline float **transpose(float **original_matrix, int rows, int cols) {
    float **transposed_matrix = malloc(cols * sizeof(float*));
    for (int i = 0; i < cols; i++) {
        transposed_matrix[i] = malloc(rows * sizeof(float));
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            transposed_matrix[j][i] = original_matrix[i][j];
        }
    }
    return transposed_matrix;
}

float sum(float **matrix, int rows, int cols) {
    float n = 0.0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            n += matrix[i][j];
        }
    }
    return n;
}

float *sum_axis_0(float **matrix, int rows, int cols) {
    float *result = malloc(cols * sizeof(float));

    for (int j = 0; j < cols; j++) {
        result[j] = 0;
        for (int i = 0; i < rows; i++) {
            result[j] += matrix[i][j];
        }
    }
    return result;
}

float mean(float *arr, int len) {
    if (len == 0) {
        return 0.0;
    }
    float sum = 0.0;
    for (int i = 0; i < len; i++) {
        sum += arr[i];
    }
    return sum / len;
}

int argmax(float *arr, int size) {
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

    for (int i = 0; i < rows; i++) {
        matrix[i] = malloc(cols * sizeof(float));
    }
    return matrix;
}
