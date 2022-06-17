//! Compression by sequence

use std::ops::Index;

use crate::{
    array2d_vec::Array2dBool,
    Len,
};



type Sequence = Vec<u32>;



#[derive(Clone, Debug, PartialEq)]
pub struct SequenceCompressed {
    vec_compressed: Sequence,
}

impl SequenceCompressed {
    pub const fn new() -> SequenceCompressed { SequenceCompressed { vec_compressed: Vec::new() } }

    const fn from(vec_compressed: Sequence) -> SequenceCompressed {
        SequenceCompressed { vec_compressed }
    }

    pub fn compress_from<A>(elements: &A) -> SequenceCompressed
    where A: Len + Index<usize, Output=bool> // this is any "array" type (Array or Vec)
    {
        let mut vec_compressed: Sequence = Vec::new();
        let mut same_in_row: u32 = 0;
        let mut bit: bool = false;
        for i in 0..elements.len() {
            // println!("i = {i}, elements[i] = {element_i}, bit = {bit}", element_i = elements[i]);
            if elements[i] == bit {
                // println!("EQ");
                same_in_row += 1;
            }
            else {
                // println!("NEQ");
                vec_compressed.push(same_in_row);
                same_in_row = 1;
                bit = !bit;
            }
            // println!("vec_compressed = {vec_compressed:?}\n");
        }
        vec_compressed.push(same_in_row);
        SequenceCompressed::from(vec_compressed)
    }

    pub fn decompress<const N: usize>(&self) -> Array2dBool<N> {
        let mut vec_decompressed: Vec<bool> = Vec::with_capacity(N);
        let mut bit: bool = false;
        for i in self.vec_compressed.iter() {
            vec_decompressed.extend(vec![bit; *i as usize]);
            bit = !bit;
        }
        Array2dBool::from_vec(vec_decompressed)
    }

    pub fn get_vec(self) -> Sequence { self.vec_compressed }

    pub fn calc_weight(&self) -> u64 {
        self.vec_compressed.iter()
            .map(|&it| it as u64)
            // TODO: map 42 = 101010 -> 6
            .map(|it| format!("{:b}", it).len() as u64 + 1)
            .sum()
    }

}

