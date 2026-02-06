use std::error::Error;
use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;

pub fn softmax_cpu(x: &mut Vec<f32>) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    for val in x.iter_mut() {
        *val = (*val - max).exp();
    }

    let sum: f32 = x.iter().sum();

    for val in x.iter_mut() {
        *val /= sum;
    }
}

pub fn softmax_gpu(x: &mut Vec<f32>) -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let ptx = Ptx::from_file("softmax.ptx");
    let module = ctx.load_module(ptx)?;
    let kernel = module.load_function("softmax")?;

    let mut x_dev = stream.memcpy_stod(x)?;

    let n = x.len();

    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0
    };

    let mut builder = stream.launch_builder(&kernel);
    builder.arg(&mut x_dev);
    builder.arg(&n);

    unsafe {builder.launch(config)}?;

    stream.synchronize()?;
    let result = stream.memcpy_dtov(&x_dev)?;
    x.copy_from_slice(&result);

    Ok(())
}
