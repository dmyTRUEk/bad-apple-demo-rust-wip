//! Utils

use std::io::Write;
use std::time::Instant;


/// [1, 0, 1] -> [true, false, true]
pub fn ua_to_ba<const N: usize>(array_u32: [u32; N]) -> [bool; N] {
    array_u32.map(|it| match it {
        0 => { false }
        1 => { true }
        _ => { panic!() }
    })
}

/// [1, 0, 1] -> vec![true, false, true]
pub fn ua_to_bv<const N: usize>(array_u32: [u32; N]) -> Vec<bool> {
    array_u32.map(|it| match it {
        0 => { false }
        1 => { true }
        _ => { panic!() }
    }).to_vec()
}



pub fn flush() {
    std::io::stdout().flush().unwrap();
}



/// Measure time of the code
///
/// let time = measure_time(|| {
///     sleep(Duration::from_secs(4));
/// });
/// println!("time = {time:.3}");
pub fn measure_time(mut f: impl FnMut()) -> f64 {
    let start = Instant::now();
    f();
    let end = Instant::now();
    (end - start).as_nanos() as f64 / 1_000_000_000_f64
}

