#ifndef LOSS_H
#define LOSS_H


void mse_loss(
    double **prediction,
    int prediction_rows,
    int prediction_cols,
    double *target,
    double **output_error,
    int regression
);

void cross_entropy_loss(
    double **prediction,
    int prediction_rows,
    int prediction_cols,
    double *target,
    double **output_error,
    int regression
);

void calc_loss(
    int loss,
    double *target,
    double **prediction,
    int prediction_rows,
    int prediction_cols,
    double **output_error,
    int regression
);

#endif
