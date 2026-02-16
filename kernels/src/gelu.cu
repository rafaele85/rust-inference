extern "C" __global__ void gelu (
    float* x,
    const int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        float val = x[i];
        float cube = val * val * val;
        float inner = sqrtf(2.0f / 3.14151965f) * (val + 0.044715f * cube);
        x[i] = val * 0.5f * (1.0f + tanhf(inner));
    }
}
