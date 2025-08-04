#ifndef FUNCTIONS_H
#define FUNCTIONS_H


void matmul(
    float **A,
    float **B,
    float **C,
    int rows_A,
    int cols_A,
    int rows_B,
    int cols_B
);

float **transpose(
    float **original_matrix,
    int rows,
    int cols
);

float sum(
    float **matrix,
    int rows,
    int cols
);

float *sum_axis_0(
    float **matrix,
    int rows,
    int cols
);

float mean(
    float *arr,
    int len
);

int argmax(
    float *arr,
    int size
);

float safe_update(
    float number,
    float max_change
);

void apply_dropout(
    float **y,
    int matrix_rows,
    int n_neurons,
    float dropout
);

float **create_matrix(
    int rows,
    int cols
);

void free_matrix(
    float **matrix
);

#endif
