#ifndef FUNCTIONS_H
#define FUNCTIONS_H


typedef struct ThreadData {
    double **A;
    double **B;
    double **C;
    int row_A_index;
    int cols_B_index;
    int rows_A;
    int cols_A;
    int rows_B;
    int cols_B;
    int startRow;
    int endRow;
} ThreadData;

void *process_row(
    void *arg
);

void matmul(
    double **A,
    double **B,
    double **C,
    int rows_A,
    int cols_A,
    int rows_B,
    int cols_B,
    int threading,
    int num_cpu
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

double **create_matrix(
    int rows,
    int cols
);

#endif
