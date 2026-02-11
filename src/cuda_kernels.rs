use std::error::Error;
use std::sync::Arc;
use cudarc::cublas::CudaBlas;
use cudarc::cudnn::result::create_handle;
use cudarc::cudnn::sys::cudnnHandle_t;
use cudarc::driver::{CudaContext, CudaFunction};
use cudarc::nvrtc::Ptx;

pub struct CudaKernels {
    pub scale: CudaFunction,
    pub extract_head: CudaFunction,
    pub combine_heads: CudaFunction,
    pub residual_add: CudaFunction,
    pub blas: CudaBlas,
    pub cudnn_handle: cudnnHandle_t,
}

impl CudaKernels {
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, Box<dyn Error>> {
        let scale_module = ctx.load_module(Ptx::from_file("scale.ptx"))?;
        let head_ops_module = ctx.load_module(Ptx::from_file("head_ops.ptx"))?;
        let residual_module = ctx.load_module(Ptx::from_file("residual.ptx"))?;
        Ok(Self {
            scale: scale_module.load_function("scale")?,
            extract_head: head_ops_module.load_function("extract_head")?,
            combine_heads: head_ops_module.load_function("combine_heads")?,
            residual_add: residual_module.load_function("residual_add")?,
            blas: CudaBlas::new(ctx.default_stream())?,
            cudnn_handle: create_handle()?
        })
    }
}