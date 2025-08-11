#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H


void relu_calc(
    float **y,
    int matrix_rows,
    int matrix_columns
);

void relu_derivative(
    float **y,
    int matrix_rows,
    int matrix_columns
);

float sigmoid(
    float x
);

void sigmoid_calc(
    float **y,
    int matrix_rows,
    int matrix_columns
);

void sigmoid_derivative(
    float **y,
    int matrix_rows,
    int matrix_columns
);

void softmax_calc(
    float **y,
    int matrix_rows,
    int matrix_columns
);

void softmax_derivative(
    float **y,
    int matrix_rows,
    int matrix_columns
);

void apply_activation_calc(
    float **y,
    int matrix_rows,
    int matrix_columns,
    int activation
);

void apply_activation_derivative(
    float **y,
    int matrix_rows,
    int matrix_columns,
    int activation
);

#endif
