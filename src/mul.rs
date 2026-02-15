use cudarc::driver::{LaunchConfig, PushKernelArg};
use crate::cuda_kernels::get_kernels;
use crate::gpu_tensor::GpuTensor;

pub fn mul(a: &mut GpuTensor, b: &GpuTensor) -> Result<(), String> {
    let kernels = get_kernels();

    let n = a.len();
    const BLOCK_SIZE: u32 = 256;
    let grid_size = (n as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;

    let config = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0
    };
    let mut builder = a.stream.launch_builder(&kernels.mul);
    builder.arg(&mut a.data);
    builder.arg(&b.data);
    builder.arg(&n);

    unsafe {builder.launch(config)};

    Ok(())
}