#ifndef FORWARD_H
#define FORWARD_H


// Тип данных для параметра потока
typedef struct ForwardData{
    float ****X_list;
    float ****Y_list;
    float ***samples;
    float ***weights;
    float **biases;
    float *layer_sizes;
    float *activations;
    float *keep_probs;
    int start_idx;
    int end_idx;
    int sample_rows;
    int sample_cols;
    int layer_sizes_rows;
    int layer_sizes_cols;
} ForwardData;

void forward(
    float **sample,
    int sample_rows,
    int sample_cols,
    float ***weights,
    float **biases,
    float ***Y,
    float *layer_sizes,
    int layer_sizes_rows,
    int layer_sizes_cols,
    float *activations
);

void *forward_worker(
    void *arg
);

void forward_threading(
    struct ForwardData forward_thread_data[],
    float ***samples,
    float ***weights,
    float **biases,
    float ****X_list,
    float ****Y_list,
    int dataset_samples_rows,
    int dataset_samples_cols,
    int dataset_samples_depth,
    float *layer_sizes,
    int layer_sizes_rows,
    int layer_sizes_cols,
    float *activations,
    float *keep_probs,
    int num_threads
);

#endif
