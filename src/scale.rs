use std::error::Error;
use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use crate::gpu_tensor::GpuTensor;

pub fn scale(x: &mut Vec<f32>, factor: f32) -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let ptx = Ptx::from_file("scale.ptx");
    let module = ctx.load_module(ptx)?;
    let kernel = module.load_function("scale")?;

    let mut x_dev = stream.memcpy_stod(x)?;
    let n = x.len() as i32;

    const BLOCK_SIZE: u32 = 256;
    let grid_size = (n as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;

    let config = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0
    };

    let mut builder = stream.launch_builder(&kernel);
    builder.arg(&mut x_dev);
    builder.arg(&factor);
    builder.arg(&n);

    unsafe {builder.launch(config)}?;

    stream.synchronize()?;
    let result = stream.memcpy_dtov(&x_dev)?;
    x.copy_from_slice(&result);

    Ok(())
}



pub fn scale_gpu(x: &mut GpuTensor, factor: f32) -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(0)?;
    let ptx = Ptx::from_file("scale.ptx");
    let module = ctx.load_module(ptx)?;
    let kernel = module.load_function("scale")?;

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
