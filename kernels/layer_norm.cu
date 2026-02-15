extern "C" __global__ void layer_norm(
    float* out,
    const float* x,
    const float* gamma,
    const float* beta,
    const int rows,
    const int cols,
    const float epsilon
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows) {
        float sum = 0.0f;

        for (int i=0; i<cols; i++) {
           sum += x[row * cols + i];
        }

        float mean = sum / cols;

        float var_sum = 0.0f;
        for (int i=0; i<cols; i++) {
           float diff = x[row * cols + i] - mean;
           var_sum += diff * diff;
        }

        float variance = var_sum / cols;
        float inv_std = 1.0f / sqrtf(variance + epsilon);

        for (int i=0; i<cols; i++) {
           float normalized = (x[row * cols + i] - mean) * inv_std;
           out[row * cols + i] = normalized * gamma[i] + beta[i];
        }
    }
}