pub fn transpose_2d(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(data.len(), rows * cols, "data.len()={} but rows*cols={}", data.len(), rows * cols);
    
    let mut out = vec![0.0; data.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}