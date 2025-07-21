#ifndef BACKWARD_H
#define BACKWARD_H


typedef struct BackwardData {
    double ***weights;
    double ****X_list;
    double ****Y_list;
    double ****grad_w_list;
    double ****grad_x_list;
    double ***grad_b_list;
    double **targets;
    double *layer_sizes;
    double *activations;
    double *epoch_losses;
    int start_idx;
    int end_idx;
    int layer_sizes_rows;
    int layer_sizes_cols;
    int matrix_rows;
    int loss;
    int regression;
    int dataset_samples_rows;
    int dataset_samples_cols;
    int dataset_targets_cols;
} BackwardData;

void *backward_worker(
    void *arg
);

void backward_threading(
    struct BackwardData backward_thread_data[],
    double ***weights,
    double **targets,
    double **biases,
    double ****X_list,
    double ****Y_list,
    double ****grad_w_list,
    double ****grad_x_list,
    double ***grad_b_list,
    double *layer_sizes,
    int layer_sizes_rows,
    int layer_sizes_cols,
    int dataset_samples_rows,
    int dataset_samples_cols,
    int dataset_targets_cols,
    int matrix_rows,
    double *activations,
    int loss,
    double *epoch_losses,
    int regression,
    int num_threads
);

#endif
