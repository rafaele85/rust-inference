//gguf.rs

use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{Read, Seek};
use crate::gguf::metadata::{read_metadata_entry, read_metadata_string, read_metadata_tensor, read_metadata_u32, read_metadata_u64, read_metadata_u64_short_array, GgufMetadataValue, TensorMetadata};
use crate::gguf::tensor_data::{bytes_to_f32, read_tensor_data_raw, read_tensor_f32, read_tensor_q4_0};

const GGUF_MAGIC: u32 = 0x46554747;



pub fn read_gguf_header(
    path: &str
) -> Result<HashMap<String, TensorMetadata>, Box<dyn Error>> {

    let mut file = File::open(path)?;

    let mut buf4 = [0; 4];
    file.read_exact(&mut buf4)?;
    let magic = u32::from_le_bytes(buf4);

    if magic != GGUF_MAGIC {
        return Err(From::from(format!("Invalid magic number: {}", magic)));
    }

    file.read_exact(&mut buf4)?;
    println!("version: {:?}", buf4);

    let tensor_count = read_metadata_u64(&mut file)?;
    println!("Tensor count: {:?}", tensor_count);

    let kv_count = read_metadata_u64(&mut file)?;
    println!("KV count: {:?}", kv_count);

    let mut headers_map: HashMap<String, GgufMetadataValue> = HashMap::new();
    for _ in 0..kv_count {
        let (key, value) = read_metadata_entry(&mut file)?;
        headers_map.insert(key, value);
    }

    print!("map size = {}", headers_map.len());

    println!("\n=== Tensors metadata ===");

    let mut tensor_metadata_map: HashMap<String, TensorMetadata> = HashMap::new();
    for _ in 0..tensor_count {
        let tensor_metadata = read_metadata_tensor(&mut file)?;
        tensor_metadata_map.insert(tensor_metadata.name.clone(), tensor_metadata);
    }

    let current_pos = file.stream_position()?;
    let alignment = 32u64;
    let data_start_offset = ((current_pos + alignment - 1) / alignment) * alignment;

    if let Some(tensor) = tensor_metadata_map.get("blk.0.attn_norm.weight") {
        let weights = read_tensor_f32(&mut file, tensor, data_start_offset)?;
        println!("first 16: {:?}", &weights[..16]);
    }

    if let Some(tensor) = tensor_metadata_map.get("blk.0.ffn_down.weight") {
        println!("Q4_0 tensor: dims={:?}, dtype={}", tensor.dims, tensor.dtype);
        let weights = read_tensor_q4_0(&mut file, tensor, data_start_offset)?;
        println!("first 8: {:?}", &weights[..8]);
        println!("values [1000..1008]: {:?}", &weights[1000..1008]);
    }

    if let Some(tensor) = tensor_metadata_map.get("output.weight") {
        let bytes = read_tensor_data_raw(&mut file, tensor, data_start_offset)?;
        let mut dequant = Vec::new();
        let ql = &bytes[0..128];
        let qh = &bytes[128..192];
        let scales = &bytes[192..208];
        for i in 0..8 {
            let lo4 = ql[i] & 0x0F;
            let qh_byte = i / 4;
            let qh_shift = (i % 4) * 2;
            let hi2 = (qh[qh_byte] >> qh_shift) & 0x03;
            let q6 = lo4 | (hi2 << 4);
            let sub_block = i / 16;
            let scale = scales[sub_block];
            let d = bytes_to_f32([bytes[208], bytes[209]]);
            let value = d * (scale as f32) * ((q6 as i32 - 32) as f32);
            dequant.push(value);
            println!("d={} value={}", d, value);
        }
        println!("first 8 {:?}", dequant);
        println!("ql[0..8]: {:?}", &ql[0..8]);
        println!("qh[0..2]: {:?}", &qh[0..2]);
    }


    Ok(tensor_metadata_map)
}