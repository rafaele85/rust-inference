use std::error::Error;
use std::sync::{Arc, OnceLock};
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
    pub gelu: CudaFunction,
    pub layer_norm: CudaFunction,
    pub embeddings: CudaFunction,
    pub rope: CudaFunction,
    pub rms_norm: CudaFunction,
    pub blas: CudaBlas,
    pub cudnn_handle: cudnnHandle_t,
}

impl CudaKernels {
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, Box<dyn Error>> {
        let scale_module = ctx.load_module(Ptx::from_file("kernels/scale.ptx"))?;
        let head_ops_module = ctx.load_module(Ptx::from_file("kernels/head_ops.ptx"))?;
        let residual_module = ctx.load_module(Ptx::from_file("kernels/residual.ptx"))?;
        let gelu_module = ctx.load_module(Ptx::from_file("kernels/gelu.ptx"))?;
        let layer_norm_module = ctx.load_module(Ptx::from_file("kernels/layer_norm.ptx"))?;
        let embedding_module = ctx.load_module(Ptx::from_file("kernels/embeddings.ptx"))?;
        let rope_module = ctx.load_module(Ptx::from_file("kernels/rope.ptx"))?;
        let rms_module = ctx.load_module(Ptx::from_file("kernels/rms.ptx"))?;
        Ok(Self {
            scale: scale_module.load_function("scale")?,
            extract_head: head_ops_module.load_function("extract_head")?,
            combine_heads: head_ops_module.load_function("combine_heads")?,
            residual_add: residual_module.load_function("residual_add")?,
            gelu: gelu_module.load_function("gelu")?,
            layer_norm: layer_norm_module.load_function("layer_norm")?,
            embeddings: embedding_module.load_function("embeddings")?,
            rope: rope_module.load_function("rope")?,
            rms_norm: rms_module.load_function("rms")?,
            blas: CudaBlas::new(ctx.default_stream())?,
            cudnn_handle: create_handle()?
        })
    }
}

unsafe impl Send for CudaKernels {}
unsafe impl Sync for CudaKernels {}

static CUDA_KERNELS: OnceLock<CudaKernels> = OnceLock::new();

pub fn get_kernels() -> &'static CudaKernels {
    CUDA_KERNELS.get_or_init(|| {
        let ctx = CudaContext::new(0).expect("Failed to create cuda context");
        CudaKernels::new(&ctx).expect("Failed to create cuda kernels")
    })
}
