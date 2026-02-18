use std::error::Error;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use crate::cuda_kernels::{get_kernels, CudaKernels};
use crate::gpu_tensor::GpuTensor;

pub fn residual_add(
    a: &mut GpuTensor,
    b: &GpuTensor
) -> Result<(), Box<dyn Error>> {
    let cuda_kernels = get_kernels();
    let n = a.len() as i32;
    const BLOCK_SIZE: usize = 256;

    let grid_size = (n as usize + BLOCK_SIZE - 1) / BLOCK_SIZE;

    let config = LaunchConfig {
        grid_dim: (grid_size as u32, 1, 1),
        block_dim: (BLOCK_SIZE as u32, 1, 1),
        shared_mem_bytes: 0
    };

    let mut builder = a.stream.launch_builder(&cuda_kernels.residual_add);
    builder.arg(&mut a.data);
    builder.arg(&b.data);
    let i32_n = n as i32;
    builder.arg(&i32_n);


    unsafe {
        builder.launch(config)
    }?;

    Ok(())
}