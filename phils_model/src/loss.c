#include <stdlib.h>
#include <math.h>
#include "loss.h"
#include "functions.h"
#include "activations.h"


///////////////////////////////////////////////////////////////////////////////
// Loss functions
///////////////////////////////////////////////////////////////////////////////

void mse_loss(double **prediction, int prediction_rows, int prediction_cols, double *target, double **output_error, int regression) {
    double **loss = malloc(prediction_rows * sizeof(double*));

    for (int i = 0; i < prediction_rows; ++i) {
        loss[i] =  malloc(prediction_cols * sizeof(double));

        int max_target_index = argmax(target, prediction_cols);
        int max_prediction_index = argmax(prediction[i], prediction_cols);

        for (int j = 0; j < prediction_cols; ++j) {
            // if (j == max_target_index && !regression) {
            //     loss[i][j] = 0.0;
            // } else {
            //     loss[i][j] = pow(target[max_target_index] - prediction[i][j], 2);
            // }

            loss[i][j] = pow(target[j] - prediction[i][j], 2);
        }
    }

    for (int i = 0; i < prediction_rows; ++i) {
        for (int j = 0; j < prediction_cols; ++j) {
            output_error[i][j] = loss[i][j];
        }
        free(loss[i]);
    }
    free(loss);
}

void cross_entropy_loss(double **prediction, int prediction_rows, int prediction_cols, double *target, double **output_error, int regression) {
    double **loss = malloc(prediction_rows * sizeof(double*));

    for (int i = 0; i < prediction_rows; ++i) {
        loss[i] =  malloc(prediction_cols * sizeof(double));

        int max_target_index = argmax(target, prediction_cols);
        int max_prediction_index = argmax(prediction[i], prediction_cols);

        for (int j = 0; j < prediction_cols; ++j) {
            // if (j == max_target_index && !regression) {
            //     loss[i][j] = 0.0;
            // } else {
            //     double p = prediction[i][j] > 1e-15 ? prediction[i][j] : 1e-15;
            //     loss[i][j] = target[max_target_index] * log(p);
            // }

            double p = prediction[i][j] > 1e-15 ? prediction[i][j] : 1e-15;
            loss[i][j] = target[j] * log(p);
        }
    }

    for (int i = 0; i < prediction_rows; ++i) {
        for (int j = 0; j < prediction_cols; ++j) {
            output_error[i][j] = loss[i][j];
        }
    }

    for (int i = 0; i < prediction_rows; ++i) {
        free(loss[i]);
    }
    free(loss);
}

///////////////////////////////////////////////////////////////////////////////

void apply_activation_calc(double **y, int matrix_rows, int matrix_columns, int activation) {
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

void apply_activation_derivative(double **y, int matrix_rows, int matrix_columns, int activation) {
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

void calc_loss(int loss, double *target, double **prediction, int prediction_rows, int prediction_cols, double **output_error, int regression) {
    if (loss == 0) {
        return mse_loss(prediction, prediction_rows, prediction_cols, target, output_error, regression);
    } else if (loss == 1) {
        return cross_entropy_loss(prediction, prediction_rows, prediction_cols, target, output_error, regression);
    }
}
