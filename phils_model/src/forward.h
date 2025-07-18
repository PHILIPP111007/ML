#ifndef FORWARD_H
#define FORWARD_H


typedef struct ForwardData {
    int dataset_index;
    double **sample;
    int sample_rows;
    int sample_cols;
    double ***weights;
    double **biases;
    double ***X;
    double ***Y;
    double *layer_sizes;
    int layer_sizes_rows;
    int layer_sizes_cols;
    double *activations;
    double keep_prob;
    int threading;
    int num_cpu;
} ForwardData;

void forward(
    double **sample,
    int sample_rows,
    int sample_cols,
    double ***weights,
    double **biases,
    double ***Y,
    double *layer_sizes,
    int layer_sizes_rows,
    int layer_sizes_cols,
    double *activations,
    int threading,
    int num_cpu
);

void *forward_train(
    void *arg
);

#endif
