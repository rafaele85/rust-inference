use std::error::Error;
use std::fs::File;
use std::io::{Read, Seek};
use std::io::SeekFrom::Start;
use crate::gguf::metadata::TensorMetadata;

pub fn read_tensor_data_raw(file: &mut File, tensor: &TensorMetadata, data_start_offset: u64) -> Result<Vec<u8>, Box<dyn Error>> {
    let num_elements: u64 = tensor.dims.iter().product();

    let byte_size = match tensor.dtype {
        0 => num_elements * 4,
        2 => (num_elements / 32) * 18,
        14 => (num_elements / 256) * 210,
        _ => return Err(format!("unknown dtype {}", tensor.dtype).into())
    };


    file.seek(Start(data_start_offset + tensor.offset))?;

    let mut buf = vec![0; byte_size as usize];
    file.read_exact(&mut buf)?;

    Ok(buf)
}

pub fn read_tensor_f32(file: &mut File, tensor: &TensorMetadata, data_start_offset: u64) -> Result<Vec<f32>, Box<dyn Error>> {
    let bytes = read_tensor_data_raw(file, tensor, data_start_offset)?;
    let values = bytes
        .chunks(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    Ok(values)
}

pub fn bytes_to_f32(bytes: [u8; 2]) -> f32 {
    half::f16::from_le_bytes(bytes).to_f32()
}

fn dequantize_block(bytes: &[u8]) -> Vec<f32> {
    let scale = bytes_to_f32([bytes[0], bytes[1]]);
    let mut dequant = vec![0.0; 32];
    for i in 0..16 {
        let q_lo = (bytes[i+2] & 0x0F) as i32 - 8;
        let q_hi = ((bytes[i+2] >> 4) & 0x0F) as i32 - 8;
        dequant[i + 0] = scale * q_lo as f32;
        dequant[i + 16] = scale * q_hi as f32;
    }

    dequant
}

pub fn read_tensor_q4_0(file: &mut File, tensor: &TensorMetadata, data_start_offset: u64) -> Result<Vec<f32>, Box<dyn Error>> {
    let bytes = read_tensor_data_raw(file, tensor, data_start_offset)?;

    let num_elements: u64 = tensor.dims.iter().product();
    let num_blocks = num_elements / 32;
    let mut out = Vec::with_capacity(num_elements as usize);
    for block_idx in 0..num_blocks {
        let block_bytes = &bytes[block_idx as usize * 18..(block_idx as usize + 1) * 18];
        out.extend(dequantize_block(block_bytes));
    }

    Ok(out)
}