#ifndef FORWARD_H
#define FORWARD_H


// Тип данных для параметра потока
typedef struct ForwardData{
    int start_idx;              // Начальный индекс выборки
    int end_idx;                // Конечный индекс выборки
    double ***samples;
    int sample_rows;
    int sample_cols;
    double ****X_list;
    double ****Y_list;
    double ***weights;
    double **biases;
    double *layer_sizes;
    int layer_sizes_rows;
    int layer_sizes_cols;       
    double *activations;
    double keep_prob;
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
    double keep_prob,
    int num_threads
);

#endif
