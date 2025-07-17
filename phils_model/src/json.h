#ifndef JSON_H
#define JSON_H


void save_weights_as_json(
    char *fname,
    double ***weights_result,
    double *layer_sizes,
    int layer_sizes_rows,
    int layer_sizes_cols
);

void save_biases_as_json(
    char *fname,
    double **biases,
    double *layer_sizes,
    int layer_sizes_rows,
    int layer_sizes_cols
);

#endif
