#include <stdlib.h>
#include <stdio.h>
#include "functions.h"
#include "forward.h"
#include "predict.h"


void *predict_thread(void *arg) {
    ThreadRange *range = (ThreadRange*)arg;

    for (int i = range->start; i < range->end; i++) {
        PredictTask *task = &range->tasks[i];

        float ***__restrict Y = malloc(task->layer_sizes_rows * sizeof(float**));
        check_if_null((float *)Y, "Y");

        // Forward pass
        forward(
            task->sample,
            task->dataset_samples_cols,
            task->dataset_samples_depth,
            task->weights,
            task->biases,
            Y,
            task->layer_sizes,
            task->layer_sizes_rows,
            task->layer_sizes_cols,
            task->activations,
            task->gpu,
            range->context,
            range->queue,
            range->program,
            range->weights_vec_buf
        );

        // Get predictions from last layer
        for (int j = 0; j < task->n_neurons_last_layer; j++) {
            int index = task->dataset_index * task->n_neurons_last_layer + j;
            task->predictions[index] = Y[task->layer_sizes_rows - 1][0][j];
        }

        // Free memory

        for (register int layer_index = 0; layer_index < task->layer_sizes_rows; layer_index++) {
            for (register int i = 0; i < task->dataset_samples_cols; i++) {
                free(Y[layer_index][i]);
            }
            free(Y[layer_index]);
        }
        free(Y);
    }

    free(range);

    return NULL;
}
