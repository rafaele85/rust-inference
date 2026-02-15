use std::error::Error;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use crate::cuda_kernels::get_kernels;
use crate::gpu_tensor::GpuTensor;

pub fn layer_norm(x: &GpuTensor, gamma: &GpuTensor, beta: &GpuTensor, epsilon: f32) -> Result<GpuTensor, Box<dyn Error>> {
    let kernels = get_kernels();

    let rows = x.shape[0] as i32;
    let cols = x.shape[1] as i32;

    let mut out = GpuTensor::alloc_zeros(x.stream.clone(), x.len(), x.shape.clone())?;

    const BLOCK_SIZE: u32 = 256;
    let grid_size = (rows as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;

    let config = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0
    };

    let mut builder = x.stream.launch_builder(&kernels.layer_norm);
    builder.arg(&mut out.data);
    builder.arg(&x.data);
    builder.arg(&gamma.data);
    builder.arg(&beta.data);
    builder.arg(&rows);
    builder.arg(&cols);
    builder.arg(&epsilon);

    unsafe {  builder.launch(config); };

    Ok((out))
}