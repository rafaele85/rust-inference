use std::error::Error;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use crate::cuda_kernels::{get_kernels, CudaKernels};
use crate::gpu_tensor::GpuTensor;

pub fn extract_head(
    input: &GpuTensor,
    seq_len: usize,
    dim: usize,
    head_dim: usize,
    head_idx: usize
) -> Result<GpuTensor, Box<dyn Error>> {
    let cuda_kernels = get_kernels();
    let kernel = &cuda_kernels.extract_head;

    let out_size = seq_len * head_dim;
    let mut out_data = input.stream.alloc_zeros::<f32>(out_size)?;
    let n = out_size as f32;
    const BLOCK_SIZE: u32 = 256;

    let grid_size = (n as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE; // ceil n / BLOCK_SIZE

    let config = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0
    };

    let mut builder = input.stream.launch_builder(&kernel);

    builder.arg(&mut out_data);
    builder.arg(&input.data);
    builder.arg(&seq_len);
    builder.arg(&dim);
    builder.arg(&head_dim);
    builder.arg(&head_idx);

    unsafe {
        builder.launch(config)?;
    }

    Ok(GpuTensor {
        data: out_data,
        stream: input.stream.clone(),
        shape: vec![seq_len, head_dim]
    })
}

pub fn combine_heads(
    output: &mut GpuTensor,
    head_out: &GpuTensor,
    seq_len: usize,
    dim: usize,
    head_dim: usize,
    head_idx: usize
) -> Result<(), Box<dyn Error>> {
    let cuda_kernels = get_kernels();
    let kernel = &cuda_kernels.combine_heads;

    let n = (seq_len * head_dim) as f32;
    const BLOCK_SIZE: u32 = 256;
    let grid_size = (n as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE; // ceil n / BLOCK_SIZE


    let config = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0
    };

    let mut builder = output.stream.launch_builder(&kernel);

    builder.arg(&mut output.data);
    builder.arg(&head_out.data);
    builder.arg(&seq_len);
    builder.arg(&dim);
    builder.arg(&head_dim);
    builder.arg(&head_idx);

    unsafe {
        builder.launch(config)?;
    }

    Ok(())
}