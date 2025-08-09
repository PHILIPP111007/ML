__kernel void adam_step_gpu(__global float *weights, __global float *grads,
                            __global float *m, __global float *v,
                            const float b1, const float b2, const float lr,
                            const float eps, const float inv_1mb1,
                            const float inv_1mb2, const float b1_minus_1,
                            const float b2_minus_1, const float max_change,
                            const int total_elements_per_sample,
                            const int dataset_samples_rows) {

  int tid = get_global_id(0);

  int total_elements = total_elements_per_sample * dataset_samples_rows;

  if (tid >= total_elements) {
    return;
  }

  int idx = tid % total_elements_per_sample;
  int sample_idx = tid / total_elements_per_sample;

  float grad = grads[idx + sample_idx * total_elements_per_sample];
  float grad_sq = grad * grad;

  m[idx + sample_idx * total_elements_per_sample] =
      b1 * m[idx + sample_idx * total_elements_per_sample] + b1_minus_1 * grad;

  v[idx + sample_idx * total_elements_per_sample] =
      b2 * v[idx + sample_idx * total_elements_per_sample] +
      b2_minus_1 * grad_sq;

  float m_hat = m[idx + sample_idx * total_elements_per_sample] * inv_1mb1;
  float v_hat = v[idx + sample_idx * total_elements_per_sample] * inv_1mb2;

  float delta = lr * m_hat / (sqrt(v_hat) + eps);

  delta = fmin(fmax(delta, -max_change), max_change);

  weights[idx + sample_idx * total_elements_per_sample] -= delta;
}
