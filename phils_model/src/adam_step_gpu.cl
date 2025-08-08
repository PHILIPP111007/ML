__kernel void adam_step_gpu(__global float *weights, __global float *grads,
                            __global float *m, __global float *v,
                            const float b1, const float b2, const float lr,
                            const float eps, const float inv_1mb1,
                            const float inv_1mb2, const float b1_minus_1,
                            const float b2_minus_1, const float max_change,
                            const int total_elements) {
  int idx = get_global_id(0);

  if (idx >= total_elements) {
    return;
  }

  float grad = grads[idx];
  float grad_sq = grad * grad;

  m[idx] = b1 * m[idx] + b1_minus_1 * grad;
  v[idx] = b2 * v[idx] + b2_minus_1 * grad_sq;

  float m_hat = m[idx] * inv_1mb1;
  float v_hat = v[idx] * inv_1mb2;
  float delta = lr * m_hat / (sqrt(v_hat) + eps);

  delta = clamp(delta, -max_change, max_change);

  weights[idx] -= delta;
}
