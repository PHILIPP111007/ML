#ifndef JSON_H
#define JSON_H


void save_weights_as_json(
    char *fname,
    float ***weights_result,
    float *layer_sizes,
    int layer_sizes_rows,
    int layer_sizes_cols
);

void save_biases_as_json(
    char *fname,
    float **biases,
    float *layer_sizes,
    int layer_sizes_rows,
    int layer_sizes_cols
);

#endif
