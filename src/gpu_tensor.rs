use std::error::Error;
use std::sync::Arc;
use cudarc::driver::{CudaSlice, CudaStream};

pub struct GpuTensor {
    pub data: CudaSlice<f32>,
    pub shape: Vec<usize>,
    pub stream: Arc<CudaStream>,
}

impl GpuTensor {
    pub fn to_cpu(&self) -> Result<Vec<f32>, Box<dyn Error>> {
        let mut result = vec![0.0; self.len()];
        self.stream.memcpy_dtoh(&self.data, &mut result)?;

        Ok(result)
    }

    pub fn from_cpu(stream: Arc<CudaStream>, data: &[f32], shape: Vec<usize>) -> Result<Self, Box<dyn Error>> {
        let data = stream.memcpy_stod(data)?;
        Ok(Self {
            stream,
            data,
            shape,
        })
    }

    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn alloc_zeros(stream: Arc<CudaStream>, len: usize, shape: Vec<usize>) -> Result<Self, Box<dyn Error>> {
        let data = stream.alloc_zeros::<f32>(len)?;
        Ok(Self { data, shape, stream })
    }
}