use std::error::Error;
use std::fs::File;
use std::io::Read;

pub enum GgufMetadataValue {
    String(String),
    U32(u32),
    F32(f32),
    ArrayString(Vec<String>),
    ArrayF32(Vec<f32>),
    ArrayU32(Vec<u32>),
    ArrayI32(Vec<i32>),
}

pub struct TensorMetadata {
    pub name: String,
    pub n_dims: u32,
    pub dims: Vec<u64>,
    pub offset: u64,
}


pub fn read_metadata_string(file: &mut File) -> Result<String, Box<dyn Error>> {
    let mut buf8 = [0; 8];
    file.read_exact(&mut buf8)?;
    let len = u64::from_le_bytes(buf8);

    let mut buf = vec![0u8; len as usize];
    file.read_exact(&mut buf)?;

    Ok(String::from_utf8(buf)?)
}

pub fn read_metadata_u32(file: &mut File) -> Result<u32, Box<dyn Error>> {
    let mut buf = [0; 4];
    file.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

pub fn read_metadata_u64(file: &mut File) -> Result<u64, Box<dyn Error>> {
    let mut buf = [0; 8];
    file.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

pub fn read_metadata_f32(file: &mut File) -> Result<f32, Box<dyn Error>> {
    let mut buf = [0; 4];
    file.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

pub fn read_metadata_i32(file: &mut File) -> Result<i32, Box<dyn Error>> {
    let mut buf = [0; 4];
    file.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

pub fn read_metadata_string_array(file: &mut File) -> Result<(Vec<String>, u64), Box<dyn Error>> {
    let elem_count = read_metadata_u64(file)?;
    let mut arr = Vec::with_capacity(elem_count as usize);
    for _ in 0..elem_count {
        let value = read_metadata_string(file)?;
        arr.push(value);
    }

    Ok((arr, elem_count))
}


pub fn read_metadata_u32_array(file: &mut File) -> Result<(Vec<u32>, u64), Box<dyn Error>> {
    let elem_count = read_metadata_u64(file)?;
    let mut arr = Vec::with_capacity(elem_count as usize);
    for _ in 0..elem_count {
        let value = read_metadata_u32(file)?;
        arr.push(value);
    }

    Ok((arr, elem_count))
}

pub fn read_metadata_u64_short_array(file: &mut File) -> Result<(Vec<u64>, u32), Box<dyn Error>> {
    let elem_count = read_metadata_u32(file)?;
    let mut arr = Vec::with_capacity(elem_count as usize);
    for _ in 0..elem_count {
        let value = read_metadata_u64(file)?;
        arr.push(value);
    }

    Ok((arr, elem_count))
}


pub fn read_metadata_i32_array(file: &mut File) -> Result<(Vec<i32>, u64), Box<dyn Error>> {
    let elem_count = read_metadata_u64(file)?;
    let mut arr = Vec::with_capacity(elem_count as usize);
    for _ in 0..elem_count {
        let value = read_metadata_i32(file)?;
        arr.push(value);
    }

    Ok((arr, elem_count))
}


pub fn read_metadata_f32_array(file: &mut File) -> Result<(Vec<f32>, u64), Box<dyn Error>> {
    let elem_count = read_metadata_u64(file)?;
    let mut arr = Vec::with_capacity(elem_count as usize);
    for _ in 0..elem_count {
        let value = read_metadata_f32(file)?;
        arr.push(value);
    }

    Ok((arr, elem_count))
}

pub fn read_metadata_unknown_array(file: &mut File) -> Result<(Vec<String>, u64), Box<dyn Error>> {
    let elem_count = read_metadata_u64(file)?;
    let mut arr = Vec::with_capacity(elem_count as usize);

    Ok((arr, elem_count))
}


pub fn read_metadata_entry(file: &mut File) -> Result<(String, GgufMetadataValue), Box<dyn Error>> {
    let key = read_metadata_string(file)?;
    let typ = read_metadata_u32(file)?;
    if (typ == 8) {
        let value = read_metadata_string(file)?;
        println!("key: {} typ: {}, value: {}", key, typ, value);
        Ok((key, GgufMetadataValue::String(value)))
    } else if typ == 4 {
        let value = read_metadata_u32(file)?;
        println!("key: {} typ: {}, value: {}", key, typ, value);
        Ok((key, GgufMetadataValue::U32(value)))
    } else if typ == 6 {
        let value = read_metadata_f32(file)?;
        println!("key: {} typ: {}, value: {}", key, typ, value);
        Ok((key, GgufMetadataValue::F32(value)))
    } else if typ == 9 {
        let elem_type = read_metadata_u32(file)?;
        if elem_type == 8 {
            let (value, elem_count) = read_metadata_string_array(file)?;
            println!("key: {} typ: {}, len {}, value: {}", key, typ, elem_count, value.iter().take(5).map(|x| x.to_string()).collect::<Vec<_>>().join(","));
            Ok((key, GgufMetadataValue::ArrayString(value)))
        } else if elem_type == 4 {
            let (value, elem_count) = read_metadata_u32_array(file)?;
            println!("key: {} typ: {}, len {}, value: {}", key, typ, elem_count, value.iter().take(5).map(|x| x.to_string()).collect::<Vec<_>>().join(","));
            Ok((key, GgufMetadataValue::ArrayU32(value)))
        } else if elem_type == 5 {
            let (value, elem_count) = read_metadata_i32_array(file)?;
            println!("key: {} typ: {}, len {}, value: {}", key, typ, elem_count, value.iter().take(5).map(|x| x.to_string()).collect::<Vec<_>>().join(","));
            Ok((key, GgufMetadataValue::ArrayI32(value)))
        } else if elem_type == 6 {
            let (value, elem_count) = read_metadata_f32_array(file)?;
            println!("key: {} typ: {}, len {}, value: {}", key, typ, elem_count, value.iter().take(5).map(|x| x.to_string()).collect::<Vec<_>>().join(","));
            Ok((key, GgufMetadataValue::ArrayF32(value)))
        } else {
            let (value, elem_count) = read_metadata_unknown_array(file)?;
            println!("key: {} typ: {}, len {}, value: {}", key, typ, elem_count, value.iter().take(5).map(|x| x.to_string()).collect::<Vec<_>>().join(","));
            println!("key: {} Unknown array of elem_type: {} elem_count {}", key, elem_type, elem_count);
            Err("unknown type".into())
        }
    } else {
        println!("key: {} typ: {}, value: {}", key, typ, "Unknown value");
        Err("unknown type".into())
    }
}

pub fn read_metadata_tensor(file: &mut File) -> Result<TensorMetadata, Box<dyn Error>> {
    let name = read_metadata_string(file)?;
    let (dims, n_dims) = read_metadata_u64_short_array(file)?;

    let dtype = read_metadata_u32(file)?;
    let offset = read_metadata_u64(file)?;

    println!("----name {} n_dims {:?} dims {:?} dtype {} offset {}", name, n_dims, dims.iter().take(5).map(|x| x.to_string()).collect::<Vec<_>>().join(","), dtype, offset);

    Ok(TensorMetadata {name, n_dims, dims, offset})
}