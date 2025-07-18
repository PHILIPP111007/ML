#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H


void relu_calc(
    double **y,
    int matrix_rows,
    int matrix_columns
);

void relu_derivative(
    double **y,
    int matrix_rows,
    int matrix_columns
);

double sigmoid(
    double x
);

void sigmoid_calc(
    double **y,
    int matrix_rows,
    int matrix_columns
);

void sigmoid_derivative(
    double **y,
    int matrix_rows,
    int matrix_columns
);

void softmax_calc(
    double **y,
    int matrix_rows,
    int matrix_columns
);

void softmax_derivative(
    double **y,
    int matrix_rows,
    int matrix_columns
);

void apply_activation_calc(
    double **y,
    int matrix_rows,
    int matrix_columns,
    int activation
);

void apply_activation_derivative(
    double **y,
    int matrix_rows,
    int matrix_columns,
    int activation
);

#endif
