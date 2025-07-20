#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "functions.h"


void matmul(double **A, double **B, double **C, int rows_A, int cols_A, int rows_B, int cols_B) {
    if (cols_A != rows_B) {
        fprintf(stderr, "Matrix dimensions mismatch: %d != %d\n", cols_A, rows_B);
        return;
    }

    for (int i = 0; i < rows_A; i++) {
        for (int k = 0; k < cols_A; k++) {
            for (int j = 0; j < cols_B; j++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

double **transpose(double **original_matrix, int rows, int cols) {
    double **transposed_matrix = (double**)malloc(cols * sizeof(double*));
    for (int i = 0; i < cols; i++) {
        transposed_matrix[i] = (double*)malloc(rows * sizeof(double));
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            transposed_matrix[j][i] = original_matrix[i][j];
        }
    }
    return transposed_matrix;
}

double sum(double **matrix, int rows, int cols) {
    double n = 0.0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            n += matrix[i][j];
        }
    }
    return n;
}

double *sum_axis_0(double **matrix, int rows, int cols) {
    double *result = malloc(cols * sizeof(double));

    for (int j = 0; j < cols; ++j) {
        result[j] = 0;
        for (int i = 0; i < rows; ++i) {
            result[j] += matrix[i][j];
        }
    }
    return result;
}

double mean(double *arr, int len) {
    if (len == 0) {
        return 0.0;
    }
    double sum = 0.0;
    for (int i = 0; i < len; ++i) {
        sum += arr[i];
    }
    return sum / len;
}

int argmax(double *arr, int size) {
    if (size <= 0) {
        return -1;
    }

    int max_idx = 0;
    for (int i = 1; i < size; ++i) {
        if (arr[i] > arr[max_idx]) {
            max_idx = i;
        }
    }
    return max_idx;
}

double safe_update(double delta, double learning_rate, double max_change) {
    double change = delta * learning_rate;
    if (change > max_change) {
        change = max_change;
    }
    else if (change < -max_change) {
        change = -max_change;
    }

    return change;
}

void dropout(double **y, int matrix_rows, int n_neurons, double keep_prob) {
    for (int i = 0; i < matrix_rows; i++) {
        for (int j = 0; j < n_neurons; j++) {
            double random = (double)rand() / RAND_MAX;
            if (random > keep_prob) {
                y[i][j] = 0.0;
            }
        }
    }
}

double **create_matrix(int rows, int cols) {
    double **matrix = malloc(rows * sizeof(double*));

    for (int i = 0; i < rows; i++) {
        matrix[i] = malloc(cols * sizeof(double));
    }
    return matrix;
}
