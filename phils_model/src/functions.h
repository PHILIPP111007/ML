#ifndef FUNCTIONS_H
#define FUNCTIONS_H


void matmul(
    double **first,
    double **second,
    double **result_matrix,
    int rows_first,
    int cols_first,
    int rows_second,
    int cols_second
);

double **transpose(
    double **original_matrix,
    int rows,
    int cols
);

double sum(
    double **matrix,
    int rows,
    int cols
);

double *sum_axis_0(
    double **matrix,
    int rows,
    int cols
);

double mean(
    double *arr,
    int len
);

int argmax(
    double *arr,
    int size
);

double safe_update(
    double delta,
    double learning_rate,
    double max_change
);

void dropout(
    double **y,
    int matrix_rows,
    int n_neurons,
    double keep_prob
);

#endif
