extern "C" __global__ void embeddings(
    float* out,
    const int* token_ids,
    const float* embed_table,
    const int seq_len,
    const int dim
) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < seq_len) {
    int token_id = token_ids[i];
    for (int d = 0; d < dim; d++) {
        out[i * dim + d] = embed_table[token_id * dim + d];
    }
  }
}