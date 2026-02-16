extern "C" __global__ void extract_head(
    float* out,
    const float* in,
    const int seq_len,
    const int dim,
    const int head_dim,
    const int head_idx
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < seq_len * head_dim) {
        int s = i / head_dim;
        int d = i % head_dim;
        out[i] = in[s * dim + head_idx * head_dim + d];
    }
}

extern "C" __global__ void combine_heads(
    float* out,
    const float* in,
    const int seq_len,
    const int dim,
    const int head_dim,
    const int head_idx
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < seq_len * head_dim) {
        int s = i / head_dim;
        int d = i % head_dim;
        out[s * dim + head_idx * head_dim + d] = in[i];
    }
}