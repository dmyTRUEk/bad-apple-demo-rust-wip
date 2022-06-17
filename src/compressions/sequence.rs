//! Compression by sequence

use std::ops::Index;

use crate::{
    array2d_vec::Array2dBool,
    extensions::Len,
    frame::{
        FRAMES_AMOUNT,
        FRAME_WH,
        Frame,
        FramesOrganisation,
        frames_to_array2d,
        load_frames,
    },
    utils::{
        flush,
        measure_time,
    },
};



type Sequence = Vec<u32>;



pub fn compress_by_sequence() {
    let frames_organisations = [
        FramesOrganisation::HNW, FramesOrganisation::HWN,
        FramesOrganisation::NHW, FramesOrganisation::NWH,
        FramesOrganisation::WHN, FramesOrganisation::WNH,
    ];

    print!("Loading frames... "); flush();
    let mut frames: Vec<Frame> = Vec::new();
    let time = measure_time(|| {
        frames = load_frames();
    });
    println!("Loaded in {time:.2} s\n");

    for frames_organisation in frames_organisations {
        println!("frames_organisation = {frames_organisation:?}");

        // compress video
        let mut sequence_compressed: SequenceCompressed = SequenceCompressed::new();
        let mut pixels: Array2dBool<{FRAME_WH*FRAMES_AMOUNT}> = Array2dBool::new(false);
        let time_spent = measure_time(|| {
            pixels = frames_to_array2d(frames.clone(), frames_organisation);
            sequence_compressed = pixels.compress_by_sequence();
        });
        let weight_uncompressed: u64 = pixels.calc_weight() as u64;
        let weight_compressed  : u64 = sequence_compressed.calc_weight();
        let ratio: f64 = weight_uncompressed as f64 / weight_compressed as f64;
        println!("ratio = {ratio:.2}\ttime_spent = {time_spent:.3} s\n");
    }
}



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





#[cfg(test)]
mod tests {

    mod compress_decompress_by_sequence {
        use super::super::*;
        use crate::utils::ua_to_ba;
        #[test]
        fn elements_false() {
            const N: usize = 10;
            let array: Array2dBool<N> = Array2dBool::new(false);
            let sequence_compressed: SequenceCompressed = array.clone().compress_by_sequence();
            let array_compressed_decompressed: Array2dBool<N> = sequence_compressed.decompress();
            let vec_compressed = sequence_compressed.get_vec();
            assert_eq!(vec![10], vec_compressed);
            assert_eq!(array, array_compressed_decompressed);
        }
        #[test]
        fn elements_true() {
            const N: usize = 10;
            let array: Array2dBool<N> = Array2dBool::new(true);
            let sequence_compressed: SequenceCompressed = array.clone().compress_by_sequence();
            let array_compressed_decompressed: Array2dBool<N> = sequence_compressed.decompress();
            let vec_compressed = sequence_compressed.get_vec();
            assert_eq!(vec![0, 10], vec_compressed);
            assert_eq!(array, array_compressed_decompressed);
        }
        #[test]
        fn elements_false_true() {
            const N: usize = 10;
            let array: Array2dBool<N> = Array2dBool::from_array(ua_to_ba([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]));
            let sequence_compressed: SequenceCompressed = array.clone().compress_by_sequence();
            let array_compressed_decompressed: Array2dBool<N> = sequence_compressed.decompress();
            let vec_compressed = sequence_compressed.get_vec();
            assert_eq!(vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1], vec_compressed);
            assert_eq!(array, array_compressed_decompressed);
        }
        #[test]
        fn elements_true_false() {
            const N: usize = 10;
            let array: Array2dBool<N> = Array2dBool::from_array(ua_to_ba([1, 0, 1, 0, 1, 0, 1, 0, 1, 0]));
            let sequence_compressed: SequenceCompressed = array.clone().compress_by_sequence();
            let array_compressed_decompressed: Array2dBool<N> = sequence_compressed.decompress();
            let vec_compressed = sequence_compressed.get_vec();
            assert_eq!(vec![0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], vec_compressed);
            assert_eq!(array, array_compressed_decompressed);
        }
        #[test]
        fn elements_4_3_2_4_1() {
            const N: usize = 14;
            let array: Array2dBool<N> = Array2dBool::from_array(ua_to_ba([
                0, 0, 0, 0,
                1, 1, 1,
                0, 0,
                1, 1, 1, 1,
                0,
            ]));
            let sequence_compressed: SequenceCompressed = array.clone().compress_by_sequence();
            let array_compressed_decompressed: Array2dBool<N> = sequence_compressed.decompress();
            let vec_compressed = sequence_compressed.get_vec();
            assert_eq!(vec![4, 3, 2, 4, 1], vec_compressed);
            assert_eq!(array, array_compressed_decompressed);
        }
        #[test]
        fn elements_0_4_3_2_4_1() {
            const N: usize = 14;
            let array: Array2dBool<N> = Array2dBool::from_array(ua_to_ba([
                1, 1, 1, 1,
                0, 0, 0,
                1, 1,
                0, 0, 0, 0,
                1,
            ]));
            let sequence_compressed: SequenceCompressed = array.clone().compress_by_sequence();
            let array_compressed_decompressed: Array2dBool<N> = sequence_compressed.decompress();
            let vec_compressed = sequence_compressed.get_vec();
            assert_eq!(vec![0, 4, 3, 2, 4, 1], vec_compressed);
            assert_eq!(array, array_compressed_decompressed);
        }
    }

}

