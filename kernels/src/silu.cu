extern "C" __global__ void silu (
    float* x,
    const int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        float val = x[i];
        x[i] = val / (1.0f + expf(-val));
    }
}
