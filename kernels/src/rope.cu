// kernels/src/rope.cu
#define ROPE_BASE_FREQ 10000.0f

extern "C" __global__ void rope(
    float* x,
    const int seq_len,
    const int dim
) {
   int half_dim = dim / 2;
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   int total = seq_len * half_dim;

   if (i < total) {
        int pos = i / half_dim;
        int pair = i % half_dim;

        float freq = 1.0f / powf(ROPE_BASE_FREQ, (2.0f * pair) / dim);
        float theta = pos * freq;

        float cos_theta = cosf(theta);
        float sin_theta = sinf(theta);

        int idx0 = pos * dim + pair;
        int idx1 = pos * dim + pair + half_dim;

        float x0 = x[idx0];
        float x1 = x[idx1];

        x[idx0] = x0 * cos_theta - x1 * sin_theta;
        x[idx1] = x1 * cos_theta + x0 * sin_theta;
   }
}