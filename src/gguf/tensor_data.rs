use std::error::Error;
use std::fs::File;
use std::io::{Read, Seek};
use std::io::SeekFrom::Start;
use half::f16;
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

pub fn dequantize_block_q6_k(bytes: &[u8]) -> Vec<f32> {
    let ql = &bytes[0..128];
    let qh = &bytes[128..192];
    let scales = &bytes[192..208];
    let d = f16::from_le_bytes([bytes[208], bytes[209]]).to_f32();

    // Convert scales to f32 (int8 values used directly)
    let scales: Vec<f32> = scales.iter().map(|&s| s as i8 as f32).collect();

    // Process ql: 128 bytes -> 256 nibbles
    // Reshape to (2, 1, 64), shift by [0, 4], AND with 0x0F, reshape to (8, 32)
    let mut ql_vals = vec![0u8; 256];
    for chunk in 0..2 {
        for i in 0..64 {
            let byte = ql[chunk * 64 + i];
            // low nibble -> group chunk*2, position i
            ql_vals[chunk * 2 * 64 + i] = byte & 0x0F;
            // high nibble -> group chunk*2 + 1, position i
            ql_vals[(chunk * 2 + 1) * 64 + i] = (byte >> 4) & 0x0F;
        }
    }
    // Reshape (4, 64) to (8, 32) by splitting each 64 into two 32s
    let ql_8_32: Vec<Vec<u8>> = (0..8).map(|g| {
        let base = (g / 2) * 64 + (g % 2) * 32;
        (0..32).map(|i| ql_vals[base + i]).collect()
    }).collect();

    // Process qh: 64 bytes -> 256 2-bit values
    // Reshape to (2, 1, 32), shift by [0, 2, 4, 6], AND with 0x03, reshape to (8, 32)
    let mut qh_vals = vec![0u8; 256];
    for chunk in 0..2 {
        for i in 0..32 {
            let byte = qh[chunk * 32 + i];
            for shift in 0..4 {
                qh_vals[chunk * 4 * 32 + shift * 32 + i] = (byte >> (shift * 2)) & 0x03;
            }
        }
    }
    // Already in (8, 32) shape

    // Combine: ql | (qh << 4), then subtract 32
    let mut q = vec![0i8; 256];
    for g in 0..8 {
        for i in 0..32 {
            let val = ql_8_32[g][i] | (qh_vals[g * 32 + i] << 4);
            q[g * 32 + i] = val as i8 - 32;
        }
    }

    // Reshape (8, 32) to (16, 16): flatten then chunk into 16s
    // Actually: (8, 32) -> each 32 is split into two 16s -> 16 groups of 16
    let mut dequant = vec![0.0f32; 256];
    for out_group in 0..16 {
        let in_group = out_group / 2;
        let offset = (out_group % 2) * 16;
        for i in 0..16 {
            let q_val = q[in_group * 32 + offset + i];
            let scale_idx = out_group;  // 0..15
            dequant[out_group * 16 + i] = (q_val as f32) * d * scales[scale_idx];
        }
    }

    dequant
}
pub fn read_tensor_q6_k(file: &mut File, tensor: &TensorMetadata, data_start_offset: u64) -> Result<Vec<f32>, Box<dyn Error>> {
    let bytes = read_tensor_data_raw(file, tensor, data_start_offset)?;

    let num_elements: u64 = tensor.dims.iter().product();
    let num_blocks = num_elements / 256;
    let mut out = Vec::with_capacity(num_elements as usize);

    for block_idx in 0..num_blocks {
        let start = block_idx as usize * 210;
        let end = start + 210;
        let block_bytes  = &bytes[start..end];
        out.extend(dequantize_block_q6_k(block_bytes));
    }


    Ok(out)
}