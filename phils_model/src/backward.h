#ifndef BACKWARD_H
#define BACKWARD_H


typedef struct BackwardData {
    int dataset_index;
    double ***weights;
    double ***Y;
    double ***X;
    double *target;
    double ***grad_w;
    double ***grad_x;
    double **grad_b;
    double *layer_sizes;
    int layer_sizes_rows;
    int layer_sizes_cols;
    int matrix_rows;
    int loss;
    double *activations;
    int threading;
    int num_cpu;
    double *epoch_losses;
    int regression;
} BackwardData;

void *backward(
    void *arg
);

#endif
