//gguf.rs

use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{Read, Seek};
use std::slice::from_raw_parts;
use crate::gguf::metadata::{read_metadata_entry, read_metadata_string, read_metadata_tensor, read_metadata_u32, read_metadata_u64, read_metadata_u64_short_array, GgufMetadataValue, TensorMetadata};
use crate::gguf::tensor_data::{read_tensor_f32, read_tensor_q4_0, read_tensor_q6_k};

const GGUF_MAGIC: u32 = 0x46554747;



pub fn read_gguf_header(
    path: &str
) -> Result<(HashMap<String, TensorMetadata>, u64), Box<dyn Error>> {

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
        println!("q4_0 first 8: {:?}", &weights[..8]);
        println!("values [1000..1008]: {:?}", &weights[1000..1008]);
    }

    if let Some(tensor) = tensor_metadata_map.get("output.weight") {
        let weights = read_tensor_q6_k(&mut file, tensor, data_start_offset)?;

        let mut f = File::open("python/ref_weights.bin").expect("Run Python script first!");
        let mut buffer = Vec::new();
        f.read_to_end(&mut buffer).unwrap();

        // Cast bytes to f32 slice
        let py_weights: &[f32] = unsafe {
            from_raw_parts(buffer.as_ptr() as *const f32, buffer.len() / 4)
        };

        println!("q6_k ({}) first 8: {:?}", tensor.dtype, &weights[..8]);
        // 3. Compare
        let mut mismatches = 0;
        for i in 0..weights.len().min(py_weights.len()) {
            let r = weights[i];
            let p = py_weights[i];
            if (r - p).abs() > 1e-3 {
                if mismatches < 5 {
                    println!("Index {}: Rust {} != Py {}", i, r, p);
                }
                mismatches += 1;
            }
        }
        println!("Total mismatches: {}", mismatches);
    }

    Ok((tensor_metadata_map, data_start_offset))
}