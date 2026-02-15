use std::error::Error;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use crate::cuda_kernels::get_kernels;
use crate::gpu_tensor::GpuTensor;

pub fn embeddings(token_ids: &[i32], embed_table: &GpuTensor) -> Result<GpuTensor, Box<dyn Error>> {
    let kernels = get_kernels();
    let seq_len = token_ids.len();
    let dim = embed_table.shape[1];

    let token_ids_dev = embed_table.stream.memcpy_stod(token_ids)?;
    let mut out = GpuTensor::alloc_zeros(embed_table.stream.clone(), seq_len * dim, vec![seq_len, dim])?;

    let BLOCK_SIZE: u32 = 256;
    let grid_size = (seq_len as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;

    let config = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0
    };

    let mut builder = embed_table.stream.launch_builder(&kernels.embeddings);
    builder.arg(&mut out.data);
    builder.arg(&token_ids_dev);
    builder.arg(&embed_table.data);
    builder.arg(&seq_len);
    builder.arg(&dim);

    unsafe { builder.launch(config) }?;

    Ok(out)
}