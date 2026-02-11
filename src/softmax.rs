use std::error::Error;
use std::ffi::c_void;
use cudarc::cudnn::result::{create_handle, create_tensor_descriptor, set_tensor4d_descriptor, softmax_forward};
use cudarc::cudnn::sys::cudnnDataType_t::CUDNN_DATA_FLOAT;
use cudarc::cudnn::sys::cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_ACCURATE;
use cudarc::cudnn::sys::cudnnSoftmaxMode_t::CUDNN_SOFTMAX_MODE_CHANNEL;
use cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW;
use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use crate::gpu_tensor::GpuTensor;

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

pub fn softmax_cudnn(x: &mut [f32]) -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let handle = create_handle()?;
    println!("Cudnn initialized");

    let desc = create_tensor_descriptor()?;
    let n = x.len() as i32;
    unsafe {
        set_tensor4d_descriptor(
            desc,
            CUDNN_TENSOR_NCHW, //format NCWH - batch, channels, height, width
            CUDNN_DATA_FLOAT,
            [1,n,1,1]
        );
    };
    println!("Cudnn descriptor created");
    let mut x_dev = stream.memcpy_stod(x)?;
    let x_ptr = x_dev.device_ptr(&stream).0;
    let x_ptr_mut = x_dev.device_ptr_mut(&stream).0;
    println!("Data copied to GPU");

    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;

    unsafe {
        softmax_forward(
            handle,
            CUDNN_SOFTMAX_ACCURATE,
            CUDNN_SOFTMAX_MODE_CHANNEL,
            &alpha as *const f32 as *const c_void,
            desc,
            x_ptr as *const c_void,
            &beta as *const f32 as *const c_void,
            desc,
            x_ptr_mut as *mut c_void,
        );
    }

    stream.synchronize()?;
    let result = stream.memcpy_dtov(&x_dev)?;
    x.copy_from_slice(&result);


    Ok(())
}


pub fn softmax_cudnn_gpu(x: &mut GpuTensor) -> Result<(), Box<dyn Error>> {
    let handle = create_handle()?;

    let desc = create_tensor_descriptor()?;
    let n = x.len() as i32;
    unsafe {
        set_tensor4d_descriptor(
            desc,
            CUDNN_TENSOR_NCHW, //format NCWH - batch, channels, height, width
            CUDNN_DATA_FLOAT,
            [1,n,1,1]
        );
    };

    let x_ptr = x.data.device_ptr(&x.stream).0;
    let x_ptr_mut = x.data.device_ptr_mut(&x.stream).0;

    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;

    unsafe {
        softmax_forward(
            handle,
            CUDNN_SOFTMAX_ACCURATE,
            CUDNN_SOFTMAX_MODE_CHANNEL,
            &alpha as *const f32 as *const c_void,
            desc,
            x_ptr as *const c_void,
            &beta as *const f32 as *const c_void,
            desc,
            x_ptr_mut as *mut c_void,
        );

    }

    Ok(())
}
