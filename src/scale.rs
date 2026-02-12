use std::error::Error;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use crate::cuda_kernels::{get_kernels, CudaKernels};
use crate::gpu_tensor::GpuTensor;

pub fn scale_gpu(x: &mut GpuTensor, factor: f32) -> Result<(), Box<dyn Error>> {
    let cuda_kernels = get_kernels();
    let kernel = &cuda_kernels.scale;

    let n = x.len() as i32;

    const BLOCK_SIZE: u32 = 256;
    let grid_size = (n as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;

    let config = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0
    };

    let mut builder = x.stream.launch_builder(&kernel);
    builder.arg(&mut x.data);
    builder.arg(&factor);
    builder.arg(&n);

    unsafe {builder.launch(config)}?;

    Ok(())
}
