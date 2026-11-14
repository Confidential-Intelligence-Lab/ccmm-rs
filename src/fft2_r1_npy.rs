use num_complex::Complex64;
use std::{fs, path::Path};

fn parse_header(header: &str, key: &str) -> String {
    let pos = header.find(key).unwrap_or_else(|| panic!("missing NPY header key: {key}"));
    header[pos + key.len()..].trim_start_matches([' ', ':']).trim().to_string()
}

pub fn read_f64_square(path: &Path, n: usize) -> Vec<Complex64> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    assert!(bytes.starts_with(b"\x93NUMPY"), "not an NPY file");
    let major = bytes[6];
    let (hlen, start) = match major {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 => (u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize, 12),
        _ => panic!("unsupported NPY version"),
    };
    let end = start + hlen;
    assert!(end <= bytes.len(), "truncated NPY header");
    let header = std::str::from_utf8(&bytes[start..end]).expect("invalid NPY header");
    let descr = parse_header(header, "'descr'");
    assert!(descr.starts_with("'<f8'") || descr.starts_with("'|f8'"), "expected little-endian float64 NPY");
    let fortran = parse_header(header, "'fortran_order'");
    assert!(fortran.starts_with("False"), "Fortran-order arrays unsupported");
    let shape = parse_header(header, "'shape'");
    let expected_shape = format!("({}, {})", n, n);
    assert!(shape.starts_with(&expected_shape), "NPY shape mismatch: expected {expected_shape}, found {shape}");
    let count = n.checked_mul(n).expect("dimension overflow");
    assert_eq!(bytes.len() - end, count * 8, "NPY payload size mismatch");
    bytes[end..].chunks_exact(8).map(|chunk| {
        let x = f64::from_le_bytes(chunk.try_into().unwrap());
        assert!(x.is_finite(), "non-finite numerical input");
        Complex64::new(x, 0.0)
    }).collect()
}

pub fn write_complex128_square(path: &Path, n: usize, values: &[Complex64]) {
    assert_eq!(values.len(), n * n);
    let dict = format!("{{'descr': '<c16', 'fortran_order': False, 'shape': ({n}, {n}), }}");
    let mut header = dict.into_bytes();
    let pad = (16 - ((10 + header.len() + 1) % 16)) % 16;
    header.extend(std::iter::repeat(b' ').take(pad));
    header.push(b'\n');
    assert!(header.len() <= u16::MAX as usize);
    let mut out = Vec::with_capacity(10 + header.len() + values.len() * 16);
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(&header);
    for z in values {
        assert!(z.re.is_finite() && z.im.is_finite(), "non-finite FFT output");
        out.extend_from_slice(&z.re.to_le_bytes());
        out.extend_from_slice(&z.im.to_le_bytes());
    }
    if let Some(parent) = path.parent() { fs::create_dir_all(parent).expect("cannot create output directory"); }
    fs::write(path, out).unwrap_or_else(|e| panic!("cannot write {}: {e}", path.display()));
}
