#ifndef FORWARD_H
#define FORWARD_H


// Тип данных для параметра потока
typedef struct ForwardData{
    double ****X_list;
    double ****Y_list;
    double ***samples;
    double ***weights;
    double **biases;
    double *layer_sizes;
    double *activations;
    double *keep_probs;
    int start_idx;
    int end_idx;
    int sample_rows;
    int sample_cols;
    int layer_sizes_rows;
    int layer_sizes_cols;
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
    double *activations
);

void *forward_worker(
    void *arg
);

void forward_threading(
    struct ForwardData forward_thread_data[],
    double ***samples,
    double ***weights,
    double **biases,
    double ****X_list,
    double ****Y_list,
    int dataset_samples_rows,
    int dataset_samples_cols,
    int dataset_samples_depth,
    double *layer_sizes,
    int layer_sizes_rows,
    int layer_sizes_cols,
    double *activations,
    double *keep_probs,
    int num_threads
);

#endif
