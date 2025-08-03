#include <stdio.h>
#include <stdlib.h>
#include "json.h"


void save_weights_as_json(char *fname, float ***weights_result, float *layer_sizes, int layer_sizes_rows, int layer_sizes_cols) {
    FILE *fp = fopen(fname, "w");

    if (!fp) {
        fprintf(stderr, "Ошибка открытия файла '%s'\n", fname);
        return;
    }

    fprintf(fp, "[");

    for (int layer_size = 0; layer_size < layer_sizes_rows; layer_size++) { 
        const int n_inputs = (int)layer_sizes[layer_size * layer_sizes_cols];
        const int n_neurons = (int)layer_sizes[layer_size * layer_sizes_cols + 1];

        fprintf(fp, "[");
        for (int i = 0; i < n_inputs; i++) {
            fprintf(fp, "[");
            for (int j = 0; j < n_neurons; j++) {
                if (j == n_neurons - 1) {
                    fprintf(fp, "%f", weights_result[layer_size][i][j]);
                } else {
                    fprintf(fp, "%f,", weights_result[layer_size][i][j]);
                }
            }

            fprintf(fp, "]");
            if (i != n_inputs - 1) {
                fprintf(fp, ",");
            }
        }

        if (layer_size != layer_sizes_rows - 1) {
            fprintf(fp, "],");
        } else {
            fprintf(fp, "]");
        }
    }
    fprintf(fp, "]");
    fclose(fp);
}

void save_biases_as_json(char *fname, float **biases, float *layer_sizes, int layer_sizes_rows, int layer_sizes_cols) {
    FILE *fp = fopen(fname, "w");

    if (!fp) {
        fprintf(stderr, "Ошибка открытия файла '%s'\\n", fname);
        return;
    }

    fprintf(fp, "[");

    for (int layer_size = 0; layer_size < layer_sizes_rows; layer_size++) { 
        const int n_neurons = (int)layer_sizes[layer_size * layer_sizes_cols + 1];

        fprintf(fp, "[");
        for (int neuron = 0; neuron < n_neurons; neuron++) {
            fprintf(fp, "[");
            fprintf(fp, "%f", biases[layer_size][neuron]);
            fprintf(fp, "]");
            if (neuron != n_neurons - 1) fprintf(fp, ",");
        }

        if (layer_size != layer_sizes_rows - 1) {
            fprintf(fp, "],");
        } else {
            fprintf(fp, "]");
        }
    }

    fprintf(fp, "]");
    fclose(fp);
}
