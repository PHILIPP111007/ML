#ifndef LOSS_H
#define LOSS_H


void mse_loss(
    float **prediction,
    int prediction_rows,
    int prediction_cols,
    float *target,
    float **output_error,
    int regression
);

void cross_entropy_loss(
    float **prediction,
    int prediction_rows,
    int prediction_cols,
    float *target,
    float **output_error,
    int regression
);

void calc_loss(
    int loss,
    float *target,
    float **prediction,
    int prediction_rows,
    int prediction_cols,
    float **output_error,
    int regression
);

#endif
