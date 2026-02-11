use std::error::Error;
use std::ffi::c_void;
use cudarc::cudnn::result::{create_tensor_descriptor, set_tensor4d_descriptor, softmax_forward};
use cudarc::cudnn::sys::cudnnDataType_t::CUDNN_DATA_FLOAT;
use cudarc::cudnn::sys::cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_ACCURATE;
use cudarc::cudnn::sys::cudnnSoftmaxMode_t::CUDNN_SOFTMAX_MODE_CHANNEL;
use cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW;
use cudarc::driver::{DevicePtr, DevicePtrMut};
use crate::cuda_kernels::CudaKernels;
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


pub fn softmax_cudnn_gpu(x: &mut GpuTensor, cuda_kernels: &CudaKernels) -> Result<(), Box<dyn Error>> {
    let desc = create_tensor_descriptor()?;
    let n = x.len() as i32;
    unsafe {
        set_tensor4d_descriptor(
            desc,
            CUDNN_TENSOR_NCHW, //format NCWH - batch, channels, height, width
            CUDNN_DATA_FLOAT,
            [1,n,1,1]
        )?;
    };

    let x_ptr = x.data.device_ptr(&x.stream).0;
    let x_ptr_mut = x.data.device_ptr_mut(&x.stream).0;

    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;

    unsafe {
        softmax_forward(
            cuda_kernels.cudnn_handle,
            CUDNN_SOFTMAX_ACCURATE,
            CUDNN_SOFTMAX_MODE_CHANNEL,
            &alpha as *const f32 as *const c_void,
            desc,
            x_ptr as *const c_void,
            &beta as *const f32 as *const c_void,
            desc,
            x_ptr_mut as *mut c_void,
        )?;

    }

    Ok(())
}


pub fn softmax_cudnn_gpu_2d(x: &mut GpuTensor, cuda_kernels: &CudaKernels) -> Result<(), Box<dyn Error>> {
    let rows = x.shape[0] as i32;
    let cols = x.shape[1] as i32;

    let desc = create_tensor_descriptor()?;

    unsafe {
        set_tensor4d_descriptor(
            desc,
            CUDNN_TENSOR_NCHW, //format NCWH - batch, channels, height, width
            CUDNN_DATA_FLOAT,
            [rows,cols,1,1]
        )?;
    };

    let x_ptr = x.data.device_ptr(&x.stream).0;
    let x_ptr_mut = x.data.device_ptr_mut(&x.stream).0;

    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;

    unsafe {
        softmax_forward(
            cuda_kernels.cudnn_handle,
            CUDNN_SOFTMAX_ACCURATE,
            CUDNN_SOFTMAX_MODE_CHANNEL,
            &alpha as *const f32 as *const c_void,
            desc,
            x_ptr as *const c_void,
            &beta as *const f32 as *const c_void,
            desc,
            x_ptr_mut as *mut c_void,
        )?;
    }

    Ok(())
}

