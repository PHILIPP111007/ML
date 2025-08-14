#include <stdlib.h>
#include <math.h>
#include "functions.h"
#include "loss.h"


///////////////////////////////////////////////////////////////////////////////
// Loss functions
///////////////////////////////////////////////////////////////////////////////

inline void mse_loss(float **__restrict prediction, int prediction_rows, int prediction_cols, float *__restrict target, float **__restrict output_error, int regression) {
    float **__restrict loss = malloc(prediction_rows * sizeof(float*));
    check_if_null((float *)loss, "loss");

    for (int i = 0; i < prediction_rows; i++) {
        loss[i] =  malloc(prediction_cols * sizeof(float));
        check_if_null((float *)loss[i], "loss[i]");

        int max_target_index = argmax(target, prediction_cols);

        for (int j = 0; j < prediction_cols; j++) {
            // if (j == max_target_index && !regression) {
            //     loss[i][j] = 0.0f;
            // } else {
            //     loss[i][j] = fast_pow((float)target[max_target_index] - prediction[i][j], 2);
            // }

            loss[i][j] = fast_pow((float)target[j] - prediction[i][j], 2);

            loss[i][j] = check_if_isnan(loss[i][j]);
        }
    }

    for (int i = 0; i < prediction_rows; i++) {
        for (int j = 0; j < prediction_cols; j++) {
            output_error[i][j] = loss[i][j];
        }
        free(loss[i]);
    }
    free(loss);
}

inline void cross_entropy_loss(float **__restrict prediction, int prediction_rows, int prediction_cols, float *__restrict target, float **__restrict output_error, int regression) {
    float **__restrict loss = malloc(prediction_rows * sizeof(float*));
    check_if_null((float *)loss, "loss");

    for (int i = 0; i < prediction_rows; i++) {
        loss[i] =  malloc(prediction_cols * sizeof(float));
        check_if_null((float *)loss[i], "loss[i]");

        int max_target_index = argmax(target, prediction_cols);

        for (int j = 0; j < prediction_cols; j++) {
            // if (j == max_target_index && !regression) {
            //     loss[i][j] = 0.0f;
            // } else {
            //     float p = prediction[i][j] > 1e-6 ? prediction[i][j] : 1e-6;
            //     loss[i][j] = target[max_target_index] * log(p);
            // }

            float p = prediction[i][j] > 1e-6 ? prediction[i][j] : 1e-6;
            loss[i][j] = target[j] * log(p);

            loss[i][j] = check_if_isnan(loss[i][j]);
        }
    }

    for (int i = 0; i < prediction_rows; i++) {
        for (int j = 0; j < prediction_cols; j++) {
            output_error[i][j] = loss[i][j];
        }
        free(loss[i]);
    }
    free(loss);
}

inline void calc_loss(int loss, float *target, float **prediction, int prediction_rows, int prediction_cols, float **output_error, int regression) {
    if (loss == 0) {
        return mse_loss(prediction, prediction_rows, prediction_cols, target, output_error, regression);
    } else if (loss == 1) {
        return cross_entropy_loss(prediction, prediction_rows, prediction_cols, target, output_error, regression);
    }
}
