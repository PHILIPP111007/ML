#include <stdlib.h>
#include "forward.h"
#include "predict.h"


void *predict_thread(void *arg) {
    ThreadRange *range = (ThreadRange*)arg;

    for (int i = range->start; i < range->end; i++) {
        PredictTask *task = &range->tasks[i];

        float ***Y = malloc(task->layer_sizes_rows * sizeof(float**));

        float **sample = malloc(task->dataset_samples_cols * sizeof(float*));
        for (int j = 0; j < task->dataset_samples_cols; j++) {
            sample[j] = malloc(task->dataset_samples_depth * sizeof(float));
            for (int k = 0; k < task->dataset_samples_depth; k++) {
                sample[j][k] = task->sample[j][k];
            }
        }

        // Forward pass
        forward(sample, task->dataset_samples_cols, task->dataset_samples_depth, task->weights, task->biases, Y, task->layer_sizes, task->layer_sizes_rows, task->layer_sizes_cols, task->activations);

        // Get predictions from last layer
        for (int j = 0; j < task->n_neurons_last_layer; j++) {
            int index = task->dataset_index * task->n_neurons_last_layer + j;
            task->predictions[index] = Y[task->layer_sizes_rows - 1][0][j];
        }

        // Free memory

        for (int layer_index = 0; layer_index < task->layer_sizes_rows; layer_index++) {
            for (int i = 0; i < task->dataset_samples_cols; i++) {
                free(Y[layer_index][i]);
            }
            free(Y[layer_index]);
        }
        free(Y);
    }
    free(range);

    return NULL;
}
