#include <stdlib.h>
#include <math.h>
#include "loss.h"
#include "functions.h"


///////////////////////////////////////////////////////////////////////////////
// Loss functions
///////////////////////////////////////////////////////////////////////////////

void mse_loss(float **prediction, int prediction_rows, int prediction_cols, float *target, float **output_error, int regression) {
    float **loss = malloc(prediction_rows * sizeof(float*));

    for (int i = 0; i < prediction_rows; ++i) {
        loss[i] =  malloc(prediction_cols * sizeof(float));

        int max_target_index = argmax(target, prediction_cols);

        for (int j = 0; j < prediction_cols; ++j) {
            if (j == max_target_index && !regression) {
                loss[i][j] = 0.0;
            } else {
                loss[i][j] = pow(target[max_target_index] - prediction[i][j], 2);
            }

            // loss[i][j] = pow(target[j] - prediction[i][j], 2);
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

void cross_entropy_loss(float **prediction, int prediction_rows, int prediction_cols, float *target, float **output_error, int regression) {
    float **loss = malloc(prediction_rows * sizeof(float*));

    for (int i = 0; i < prediction_rows; ++i) {
        loss[i] =  malloc(prediction_cols * sizeof(float));

        int max_target_index = argmax(target, prediction_cols);

        for (int j = 0; j < prediction_cols; ++j) {
            if (j == max_target_index && !regression) {
                loss[i][j] = 0.0;
            } else {
                float p = prediction[i][j] > 1e-15 ? prediction[i][j] : 1e-15;
                loss[i][j] = target[max_target_index] * log(p);
            }

            // float p = prediction[i][j] > 1e-15 ? prediction[i][j] : 1e-15;
            // loss[i][j] = target[j] * log(p);
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

void calc_loss(int loss, float *target, float **prediction, int prediction_rows, int prediction_cols, float **output_error, int regression) {
    if (loss == 0) {
        return mse_loss(prediction, prediction_rows, prediction_cols, target, output_error, regression);
    } else if (loss == 1) {
        return cross_entropy_loss(prediction, prediction_rows, prediction_cols, target, output_error, regression);
    }
}
