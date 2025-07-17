#include <stdlib.h>
#include <math.h>
#include "activations.h"


///////////////////////////////////////////////////////////////////////////////
// Activation functions
///////////////////////////////////////////////////////////////////////////////

void relu_calc(double **y, int matrix_rows, int matrix_columns) {
    for (int i = 0; i < matrix_rows; ++i) {
        for (int j = 0; j < matrix_columns; ++j) {
            if (y[i] > 0) {
                y[i][j] = y[i][j];
            }
            else {
                y[i][j] = 0.0;
            }
        }
    }
}

void relu_derivative(double **y, int matrix_rows, int matrix_columns) {
    for (int i = 0; i < matrix_rows; ++i) {
        for (int j = 0; j < matrix_columns; ++j) {
            if (y[i] > 0) {
                y[i][j] = y[i][j];
            }
            else {
                y[i][j] = 0.0;
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////////

double sigmoid(double x) {
    double n = exp(x);
    if (x >= 0) {
        return 1.0 / (1.0 + n);
    } else {
        return n / (1.0 + n);
    }
}

void sigmoid_calc(double **y, int matrix_rows, int matrix_columns) {
    for (int i = 0; i < matrix_rows; ++i) {
        for (int j = 0; j < matrix_columns; ++j) {
            y[i][j] = sigmoid(y[i][j]);
        }
    }
}

void sigmoid_derivative(double **y, int matrix_rows, int matrix_columns) {
    double **f = malloc(matrix_rows * sizeof(double*));

    for (int i = 0; i < matrix_rows; ++i) {
        f[i] = malloc(matrix_columns * sizeof(double));
        for (int j = 0; j < matrix_columns; ++j) {
            f[i][j] = sigmoid(y[i][j]);
        }
    }

    for (int i = 0; i < matrix_rows; ++i) {
        for (int j = 0; j < matrix_columns; ++j) {
            y[i][j] = f[i][j] * (1.0 - f[i][j]);
        }
    }
}

///////////////////////////////////////////////////////////////////////////////

// Метод softmax (возвращает нормализованные вероятности классов).
void softmax_calc(double **y, int matrix_rows, int matrix_columns) {
    double max_val = y[0][0];

    for (int i = 1; i < matrix_rows; ++i) {
        for (int j = 1; j < matrix_columns; ++j) {
            if (y[i][j] > max_val) {
                max_val = y[i][j];
            }
        }
    }

    // Отнимем максимум от каждого элемента для стабилизации экспоненты.
    double sum_exp = 0.0;
    for (int i = 0; i < matrix_rows; ++i) {
        for (int j = 0; j < matrix_columns; ++j) {
            y[i][j] = exp(y[i][j] - max_val);
            sum_exp += y[i][j];
        }
    }

    // Нормализация путем деления каждого элемента на сумму экспонент.
    for (int i = 0; i < matrix_rows; ++i) {
        for (int j = 0; j < matrix_columns; ++j) {
            y[i][j] /= sum_exp;
        }
    }
}

void softmax_derivative(double **y, int matrix_rows, int matrix_columns) {
    for (int i = 0; i < matrix_rows; ++i) {
        for (int j = 0; j < matrix_columns; ++j) {
            y[i][j] = y[i][j];
        }
    }
}
