#include <stdlib.h>
#include <math.h>
#include "functions.h"
#include "activations.h"


///////////////////////////////////////////////////////////////////////////////
// Activation functions
///////////////////////////////////////////////////////////////////////////////

inline void relu_calc(float **y, int matrix_rows, int matrix_columns) {
    for (int i = 0; i < matrix_rows; i++) {
        for (int j = 0; j < matrix_columns; j++) {
            if (y[i] <= 0) {
                y[i][j] = 0.0;
            }
        }
    }
}

inline void relu_derivative(float **y, int matrix_rows, int matrix_columns) {
    for (int i = 0; i < matrix_rows; i++) {
        for (int j = 0; j < matrix_columns; j++) {
            if (y[i] <= 0) {
                y[i][j] = 0.0;
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////////

inline float sigmoid(float x) {
    float n = exp(x);
    if (x >= 0) {
        return 1.0 / (1.0 + n);
    } else {
        return n / (1.0 + n);
    }
}

inline void sigmoid_calc(float **y, int matrix_rows, int matrix_columns) {
    for (int i = 0; i < matrix_rows; i++) {
        for (int j = 0; j < matrix_columns; j++) {
            y[i][j] = sigmoid(y[i][j]);
        }
    }
}

inline void sigmoid_derivative(float **y, int matrix_rows, int matrix_columns) {
    float **f = create_matrix(matrix_rows, matrix_columns);

    for (int i = 0; i < matrix_rows; i++) {
        for (int j = 0; j < matrix_columns; j++) {
            f[i][j] = sigmoid(y[i][j]);
        }
    }

    for (int i = 0; i < matrix_rows; i++) {
        #pragma omp simd
        for (int j = 0; j < matrix_columns; j++) {
            y[i][j] = f[i][j] * (1.0 - f[i][j]);
        }
    }

    free_matrix(f);
}

///////////////////////////////////////////////////////////////////////////////

// The softmax method (returns normalized class probabilities)
inline void softmax_calc(float **y, int matrix_rows, int matrix_columns) {
    float max_val = y[0][0];

    for (int i = 1; i < matrix_rows; i++) {
        for (int j = 1; j < matrix_columns; j++) {
            if (y[i][j] > max_val) {
                max_val = y[i][j];
            }
        }
    }

    // Let's subtract the maximum from each element to stabilize the exponent
    float sum_exp = 0.0;
    for (int i = 0; i < matrix_rows; i++) {
        for (int j = 0; j < matrix_columns; j++) {
            y[i][j] = exp(y[i][j] - max_val);
            sum_exp += y[i][j];
        }
    }

    // Normalization by dividing each element by the sum of the exponents
    for (int i = 0; i < matrix_rows; i++) {
        for (int j = 0; j < matrix_columns; j++) {
            y[i][j] /= sum_exp;
        }
    }
}

inline void softmax_derivative(float **y, int matrix_rows, int matrix_columns) {
    for (int i = 0; i < matrix_rows; i++) {
        for (int j = 0; j < matrix_columns; j++) {
            y[i][j] = y[i][j];
        }
    }
}

///////////////////////////////////////////////////////////////////////////////

inline void apply_activation_calc(float **y, int matrix_rows, int matrix_columns, int activation) {
    if (activation == 0) {
        relu_calc(y, matrix_rows, matrix_columns);
    } else if (activation == 1) {
        sigmoid_calc(y, matrix_rows, matrix_columns);
    } else if (activation == 2) {
        softmax_calc(y, matrix_rows, matrix_columns);
    } else if (activation == 3) {
        return;
    }
}

inline void apply_activation_derivative(float **y, int matrix_rows, int matrix_columns, int activation) {
    if (activation == 0) {
        relu_derivative(y, matrix_rows, matrix_columns);
    } else if (activation == 1) {
        sigmoid_derivative(y, matrix_rows, matrix_columns);
    } else if (activation == 2) {
        softmax_derivative(y, matrix_rows, matrix_columns);
    } else if (activation == 3) {
        return;
    }
}
