use std::error::Error;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use crate::cuda_kernels::get_kernels;
use crate::gpu_tensor::GpuTensor;

pub fn rope(x: &mut GpuTensor) -> Result<(), Box<dyn Error>> {
    let cuda_kernels = get_kernels();

    let seq_len = x.shape[0];
    let dim = x.shape[1];
    const BLOCK_SIZE: u32 = 256;
    let half_dim = dim / 2;
    let total_work = seq_len * half_dim;
    let grid_size = (total_work as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;

    let config = LaunchConfig {
        grid_dim: (grid_size as u32, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = x.stream.launch_builder(&cuda_kernels.rope);
    builder.arg(&mut x.data);
    builder.arg(&seq_len);
    builder.arg(&dim);

    unsafe { builder.launch(config) }?;

    Ok(())
}