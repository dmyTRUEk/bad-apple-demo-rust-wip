//! Huffman compression

use std::{
    collections::HashMap,
    fmt::Debug,
    hash::Hash,
    ops::Index,
};

use crate::{
    array2d_vec::Array2dBool,
    extensions::{
        HashMapExtensionInvert,
        HashMapExtensionSetOneOrIncreaseByOne,
        Len,
        VecExtensionAdd,
        VecExtensionSorted,
    },
    frame::{
        FRAMES_AMOUNT,
        FRAME_WH,
        FramesOrganisation,
        frames_to_array2d,
        load_frames,
    },
    utils::{
        flush,
        measure_time,
    },
};



pub type WordUncompressed = Vec<bool>;
pub type WordCompressed   = Vec<bool>;

pub type VecCompressed  = Vec<bool>;
pub type DictCompress   = HashMap<WordUncompressed, WordCompressed>;
pub type DictDecompress = HashMap<WordCompressed, WordUncompressed>;



pub fn compress_video_by_huffman_with_word_len(
    pixels: &Array2dBool<{FRAME_WH*FRAMES_AMOUNT}>,
    word_len: usize
) -> HuffmanCompressed {
    let mut huffman_compressed: HuffmanCompressed = HuffmanCompressed::new();
    let time_spent = measure_time(|| {
        huffman_compressed = pixels.compress_by_huffman(word_len);
    });
    let weight_uncompressed: u64 = pixels.calc_weight() as u64;
    let weight_compressed  : u64 = huffman_compressed.calc_weight();
    let ratio: f64 = weight_uncompressed as f64 / weight_compressed as f64;
    // println!("uncompressed weight = {weight_uncompressed}");
    // println!("compressed   weight = {weight_compressed}");
    // println!("ratio = {ratio}");
    println!("word_len = {word_len}\tratio = {ratio}\ttime_spent = {time_spent:.3} s");
    huffman_compressed
}

pub fn compress_video_by_huffman() {
    // GCD(480, 360) = 10, 20, 30, 40, 60, 120
    let word_lens: Vec<usize> = vec![
        10, 20, 30, 40, 50, 60, 80, 100, 120,
        240, 360, 480, 600, 1200, 1800,
        // 3600, 4800, 6000,
        // 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500,
        // 12000, 24000, 36000, 48000, 60000,
    ].add(
        vec![
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            // 10, 20, 30, 40, 50, 60, 70, 80, 90,
        ].iter().map(|it| it * FRAMES_AMOUNT).collect::<Vec<usize>>()
    ).sorted();

    // HNW, HWN, NHW, NWH, WHN, WNH
    // let frames_organisation: FramesOrganisation = FramesOrganisation::HNW;
    // let frames_organisation: FramesOrganisation = FramesOrganisation::HWN;
    // let frames_organisation: FramesOrganisation = FramesOrganisation::NHW;
    // let frames_organisation: FramesOrganisation = FramesOrganisation::NWH;
    // let frames_organisation: FramesOrganisation = FramesOrganisation::WHN;
    let frames_organisation: FramesOrganisation = FramesOrganisation::WNH;
    println!("frames_organisation = {frames_organisation:?}");

    print!("\nLoading frames... "); flush();
    let mut pixels: Array2dBool<{FRAME_WH*FRAMES_AMOUNT}> = Array2dBool::new(false);
    let time = measure_time(|| {
        let frames = load_frames();
        pixels = frames_to_array2d(frames, frames_organisation);
    });
    println!("Loaded in {time:.2} s\n");

    // compress video
    for word_len in word_lens {
        compress_video_by_huffman_with_word_len(&pixels, word_len);
        // println!();
    }
}



#[derive(Clone, Debug, PartialEq)]
pub struct HuffmanCompressed {
    vec_compressed: VecCompressed,
    dict_decompress: DictDecompress,
}

impl HuffmanCompressed {
    pub fn new() -> HuffmanCompressed { HuffmanCompressed { vec_compressed: Vec::new(), dict_decompress: HashMap::new() } }

    const fn from(vec_compressed: VecCompressed, dict_decompress: DictDecompress) -> HuffmanCompressed {
        HuffmanCompressed { vec_compressed, dict_decompress }
    }

    pub fn compress_from<A>(elements: &A, word_len: usize) -> HuffmanCompressed
    where A: Len + Index<usize, Output=bool> // this is any "array" type (Array or Vec)
    {
        let n: usize = elements.len();
        let mut words_freq: HashMap<WordUncompressed, u32> = HashMap::new();

        for i in (0..n).step_by(word_len) {
            let mut word: Vec<bool> = Vec::with_capacity(word_len);
            for j in 0..word_len {
                word.push(if i+j < n {elements[i+j]} else {false});
            }
            // println!("word = {word}", word = word.to_01_string());
            assert_eq!(word_len, word.len());
            words_freq.set_or_inc(word);
        }
        // println!("map = {words_freq:?}\n\n");

        let dict_compress: DictCompress = build_best_dictionary(words_freq.into_iter().collect());

        let mut vec_compressed: VecCompressed = vec![];
        for i in (0..n).step_by(word_len) {
            let mut word: Vec<bool> = Vec::with_capacity(word_len);
            for j in 0..word_len {
                word.push(if i+j < n {elements[i+j]} else {false});
            }
            // println!("word = {word}", word = word.to_01_string()));
            assert_eq!(word_len, word.len());
            let word_compressed: Vec<bool> = dict_compress.get(&word).unwrap().clone();
            // println!("word_compressed = {word_compressed}", word_compressed = word_compressed.to_01_string()));
            vec_compressed.extend(word_compressed);
        }

        let dict_decompress = dict_compress.invert();
        HuffmanCompressed::from(vec_compressed, dict_decompress)
    }

    pub fn decompress<const N: usize>(&self) -> Array2dBool<N> {
        let vec_compressed = &self.vec_compressed;
        let dict_decompress = &self.dict_decompress;
        let mut vec_decompressed: Vec<bool> = Vec::with_capacity(N);
        let mut word_compressed: WordCompressed = vec![];
        for &bit in vec_compressed {
            word_compressed.push(bit);
            if dict_decompress.contains_key(&word_compressed) {
                let word_uncompressed: WordUncompressed = dict_decompress.get(&word_compressed).unwrap().clone();
                vec_decompressed.extend(word_uncompressed);
                word_compressed = vec![];
            }
        }
        assert_eq!(N, vec_decompressed.len());
        // let mut res_array: Array2dBool<N> = Array2dBool::new(false);
        // for i in 0..vec_decompressed.len() {
        //     res_array.elements[i] = vec_decompressed[i];
        // }
        // res_array;
        Array2dBool::from_vec(vec_decompressed)
    }

    // pub fn get_vec(self) -> VecCompressed { self.vec_compressed }
    // pub fn get_dict(self) -> DictDecompress { self.dict_decompress }
    /// get VecCompressed and DictDecompress
    pub fn get_vec_and_dict(self) -> (VecCompressed, DictDecompress) { (self.vec_compressed, self.dict_decompress) }

    pub fn calc_weight(&self) -> u64 {
        // let mut res: u64 = 0;
        // res += self.vec_compressed.len() as u64;
        // for (key, value) in &self.dict_decompress {
        //     res += key.len() as u64 + value.len() as u64 + 1;
        // }
        // res
        self.vec_compressed.len() as u64 + self.dict_decompress.iter()
                .map(|(key, value)| key.len() as u64 + value.len() as u64 + 1)
                .sum::<u64>()
    }

}



pub fn build_best_dictionary<T>(words_freq: Vec<(T, u32)>) -> HashMap<T, WordCompressed>
where T : Clone + Debug + Eq + Hash
{
    // println!("CALL");
    // println!("words_freq.len() = {words_freq_len}", words_freq_len = words_freq.len());
    assert!(words_freq.iter().all(|(_, v)| v > &0));
    let words_freq_best = match words_freq.len() {
        0 => { HashMap::new() }
        1 => {
            HashMap::from([(words_freq[0].0.clone(), vec![false])])
        }
        2 => {
            let mut words_freq_iter = words_freq.into_iter();
            let (key_a, value_a): (T, u32) = words_freq_iter.next().unwrap();
            let (key_b, value_b): (T, u32) = words_freq_iter.next().unwrap();
            let (key_0, key_1): (T, T) = if value_a >= value_b { (key_a, key_b) } else { (key_b, key_a) };
            HashMap::from([
                (key_0, vec![false]),
                (key_1, vec![true ]),
            ])
        }
        _ => {
            let mut words_freq_separated: Vec<(T, u32)> = words_freq.into_iter().collect();
            words_freq_separated.sort_by(|a, b| b.1.cmp(&a.1));
            // let words_freq_sorted: Vec<(T, u32)> = vec![];
            let mut words_freq_l: Vec<(T, u32)> = Vec::new();
            let mut words_freq_r: Vec<(T, u32)> = Vec::new();
            for (key, value) in words_freq_separated {
                let sum_l: u32 = words_freq_l.iter().map(|(_, v)| v).sum();
                let sum_r: u32 = words_freq_r.iter().map(|(_, v)| v).sum();
                if sum_l <= sum_r {
                    words_freq_l.push((key, value));
                }
                else {
                    words_freq_r.push((key, value));
                }
            }
            // println!("words_freq_l = {words_freq_l:?}");
            // println!("words_freq_r = {words_freq_r:?}");

            let words_freq_l_best: HashMap<T, WordCompressed> = build_best_dictionary(words_freq_l);
            let words_freq_r_best: HashMap<T, WordCompressed> = build_best_dictionary(words_freq_r);
            // println!("words_freq_l_best = {words_freq_l_best:?}");
            // println!("words_freq_r_best = {words_freq_r_best:?}");

            // println!("joining...");
            let mut words_freq_best: HashMap<T, WordCompressed> = HashMap::new();
            if words_freq_l_best.len() == 1 {
                let (key, _value) = words_freq_l_best.iter().next().unwrap();
                words_freq_best.insert(key.clone(), vec![false]);
            }
            else {
                for (key, value) in &words_freq_l_best {
                    words_freq_best.insert(key.clone(), vec![false].add(value.clone()));
                }
            }

            if words_freq_r_best.len() == 1 {
                let (key, _value) = words_freq_r_best.iter().next().unwrap();
                words_freq_best.insert(key.clone(), vec![true]);
            }
            else {
                for (key, value) in &words_freq_r_best {
                    words_freq_best.insert(key.clone(), vec![true].add(value.clone()));
                }
            }
            // println!("joined: {words_freq_best:?}\n");

            words_freq_best
        }
    };
    // println!("returning: {words_freq_best:?}");
    words_freq_best
}





#[cfg(test)]
mod tests {

    mod build_best_dictionary {
        use std::collections::HashMap;
        use super::super::{build_best_dictionary, WordCompressed};
        use crate::utils::ua_to_bv;

        #[test]
        fn dict_size_0() {
            let dict: Vec<(&str, u32)> = vec![];
            let expected: HashMap<&str, WordCompressed> = HashMap::new();
            let actual  : HashMap<&str, WordCompressed> = build_best_dictionary(dict);
            assert_eq!(expected, actual);
        }

        #[test]
        fn dict_size_1() {
            let a = "a";
            let dict: Vec<(&str, u32)> = vec![(a.clone(), 100)];
            let expected: HashMap<&str, WordCompressed> = HashMap::from([(a, ua_to_bv([0]))]);
            let actual  : HashMap<&str, WordCompressed> = build_best_dictionary(dict);
            assert_eq!(expected, actual);
        }

        #[test]
        fn dict_size_2() {
            let (a, b) = ("a", "b");
            let dict: Vec<(&str, u32)> = vec![
                (a.clone(), 100),
                (b.clone(), 99),
            ];
            let expected: HashMap<&str, WordCompressed> = HashMap::from([
                (a, ua_to_bv([0])),
                (b, ua_to_bv([1])),
            ]);
            let actual  : HashMap<&str, WordCompressed> = build_best_dictionary(dict);
            assert_eq!(expected, actual);
        }

        // #[ignore]
        // #[test]
        // fn build_best_dictionary_3_0_0_0() {
        //     let (a, b, c) = ("a", "b", "c");
        //     let dict: Vec<(&str, u32)> = vec![
        //         (a.clone(), 0),
        //         (b.clone(), 0),
        //         (c.clone(), 0),
        //     ];
        //     let expected: HashMap<&str, WordCompressed> = HashMap::from([
        //         (a, vec![false]),
        //         (b, vec![false, true]),
        //         (c, vec![true , true]),
        //     ]);
        //     let actual  : HashMap<&str, WordCompressed> = build_best_dictionary(dict);
        //     assert_eq!(expected, actual);
        // }

        #[test]
        fn dict_size_3_1_1_1() {
            let (a, b, c) = ("a", "b", "c");
            let dict: Vec<(&str, u32)> = vec![
                (a.clone(), 1),
                (b.clone(), 1),
                (c.clone(), 1),
            ];
            let expected: HashMap<&str, WordCompressed> = HashMap::from([
                (a, ua_to_bv([0, 0])),
                (b, ua_to_bv([1])),
                (c, ua_to_bv([0, 1])),
            ]);
            let actual  : HashMap<&str, WordCompressed> = build_best_dictionary(dict);
            assert_eq!(expected, actual);
        }

        #[test]
        fn dict_size_3_100_50_50() {
            let (a, b, c) = ("a", "b", "c");
            let dict: Vec<(&str, u32)> = vec![
                (a.clone(), 100),
                (b.clone(), 50),
                (c.clone(), 50),
            ];
            let expected: HashMap<&str, WordCompressed> = HashMap::from([
                (a, ua_to_bv([0])),
                (b, ua_to_bv([1, 0])),
                (c, ua_to_bv([1, 1])),
            ]);
            let actual  : HashMap<&str, WordCompressed> = build_best_dictionary(dict);
            assert_eq!(expected, actual);
        }

        #[test]
        fn dict_size_3_50_100_50() {
            let (a, b, c) = ("a", "b", "c");
            let dict: Vec<(&str, u32)> = vec![
                (a.clone(), 50),
                (b.clone(), 100),
                (c.clone(), 50),
            ];
            let expected: HashMap<&str, WordCompressed> = HashMap::from([
                (a, ua_to_bv([1, 0])),
                (b, ua_to_bv([0])),
                (c, ua_to_bv([1, 1])),
            ]);
            let actual  : HashMap<&str, WordCompressed> = build_best_dictionary(dict);
            assert_eq!(expected, actual);
        }

        #[test]
        fn dict_size_3_50_50_100() {
            let (a, b, c) = ("a", "b", "c");
            let dict: Vec<(&str, u32)> = vec![
                (a.clone(), 50),
                (b.clone(), 50),
                (c.clone(), 100),
            ];
            let expected: HashMap<&str, WordCompressed> = HashMap::from([
                (a, ua_to_bv([1, 0])),
                (b, ua_to_bv([1, 1])),
                (c, ua_to_bv([0])),
            ]);
            let actual  : HashMap<&str, WordCompressed> = build_best_dictionary(dict);
            assert_eq!(expected, actual);
        }

        #[test]
        fn dict_size_5() {
            let (a, b, c, d, e) = ("a", "b", "c", "d", "e");
            let dict: Vec<(&str, u32)> = vec![
                (a.clone(), 100),
                (b.clone(), 25),
                (c.clone(), 25),
                (d.clone(), 25),
                (e.clone(), 25),
            ];
            let expected: HashMap<&str, WordCompressed> = HashMap::from([
                (a, ua_to_bv([0])),
                (b, ua_to_bv([1, 0, 0])),
                (c, ua_to_bv([1, 1, 0])),
                (d, ua_to_bv([1, 0, 1])),
                (e, ua_to_bv([1, 1, 1])),
            ]);
            let actual  : HashMap<&str, WordCompressed> = build_best_dictionary(dict);
            println!("expected = {expected:#?}");
            println!("actual   = {actual:#?}");
            assert_eq!(expected, actual);
        }

    }



    mod compress_decompress_by_huffman {
        mod array_size_10 {
            mod word_len_1 {
                use std::collections::HashMap;
                use super::super::super::super::*;
                use crate::utils::ua_to_bv;
                #[test]
                fn elements_false() {
                    const N: usize = 10;
                    const WORD_LEN: usize = 1;
                    let array: Array2dBool<N> = Array2dBool::new(false);
                    let huffman_compressed: HuffmanCompressed = array.clone().compress_by_huffman(WORD_LEN);
                    let array_compressed_decompressed: Array2dBool<N> = huffman_compressed.decompress();
                    let (vec_compressed, dict_decompress) = huffman_compressed.get_vec_and_dict();
                    assert_eq!(ua_to_bv([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), vec_compressed);
                    assert_eq!(HashMap::from([(ua_to_bv([0]), ua_to_bv([0]))]), dict_decompress);
                    assert_eq!(array, array_compressed_decompressed);
                }
                #[test]
                fn elements_true() {
                    const N: usize = 10;
                    const WORD_LEN: usize = 1;
                    let array: Array2dBool<N> = Array2dBool::new(true);
                    let huffman_compressed: HuffmanCompressed = array.clone().compress_by_huffman(WORD_LEN);
                    let array_compressed_decompressed: Array2dBool<N> = huffman_compressed.decompress();
                    let (vec_compressed, dict_decompress) = huffman_compressed.get_vec_and_dict();
                    assert_eq!(ua_to_bv([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), vec_compressed);
                    assert_eq!(HashMap::from([(ua_to_bv([0]), ua_to_bv([1]))]), dict_decompress);
                    assert_eq!(array, array_compressed_decompressed);
                }
            }

            mod word_len_2 {
                use std::collections::HashMap;
                use super::super::super::super::*;
                use crate::utils::{ua_to_ba, ua_to_bv};
                #[test]
                fn elements_false_false() {
                    const N: usize = 10;
                    const WORD_LEN: usize = 2;
                    let array: Array2dBool<N> = Array2dBool::new(false);
                    let huffman_compressed: HuffmanCompressed = array.clone().compress_by_huffman(WORD_LEN);
                    let array_compressed_decompressed: Array2dBool<N> = huffman_compressed.decompress();
                    let (vec_compressed, dict_decompress) = huffman_compressed.get_vec_and_dict();
                    assert_eq!(ua_to_bv([0, 0, 0, 0, 0]), vec_compressed);
                    assert_eq!(HashMap::from([(ua_to_bv([0]), ua_to_bv([0, 0]))]), dict_decompress);
                    assert_eq!(array, array_compressed_decompressed);
                }
                #[test]
                fn elements_false_true() {
                    const N: usize = 10;
                    const WORD_LEN: usize = 2;
                    let array: Array2dBool<N> = Array2dBool::from_array(ua_to_ba([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]));
                    let huffman_compressed: HuffmanCompressed = array.clone().compress_by_huffman(WORD_LEN);
                    let array_compressed_decompressed: Array2dBool<N> = huffman_compressed.decompress();
                    let (vec_compressed, dict_decompress) = huffman_compressed.get_vec_and_dict();
                    assert_eq!(ua_to_bv([0, 0, 0, 0, 0]), vec_compressed);
                    assert_eq!(HashMap::from([(ua_to_bv([0]), ua_to_bv([0, 1]))]), dict_decompress);
                    assert_eq!(array, array_compressed_decompressed);
                }
            }
        }

        mod array_size_12 {
            mod word_len_3 {
                use std::collections::HashMap;
                use super::super::super::super::*;
                use crate::utils::{ua_to_ba, ua_to_bv};
                #[test]
                fn elements_false_false_false() {
                    const N: usize = 12;
                    const WORD_LEN: usize = 3;
                    let array: Array2dBool<N> = Array2dBool::new(false);
                    let huffman_compressed: HuffmanCompressed = array.clone().compress_by_huffman(WORD_LEN);
                    let array_compressed_decompressed: Array2dBool<N> = huffman_compressed.decompress();
                    let (vec_compressed, dict_decompress) = huffman_compressed.get_vec_and_dict();
                    assert_eq!(ua_to_bv([0, 0, 0, 0]), vec_compressed);
                    assert_eq!(HashMap::from([(ua_to_bv([0]), ua_to_bv([0, 0, 0]))]), dict_decompress);
                    assert_eq!(array, array_compressed_decompressed);
                }
                #[test]
                fn elements_false_false_true() {
                    const N: usize = 12;
                    const WORD_LEN: usize = 3;
                    let array: Array2dBool<N> = Array2dBool::from_array(ua_to_ba([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]));
                    let huffman_compressed: HuffmanCompressed = array.clone().compress_by_huffman(WORD_LEN);
                    let array_compressed_decompressed: Array2dBool<N> = huffman_compressed.decompress();
                    let (vec_compressed, dict_decompress) = huffman_compressed.get_vec_and_dict();
                    assert_eq!(ua_to_bv([0, 0, 0, 0]), vec_compressed);
                    assert_eq!(HashMap::from([(ua_to_bv([0]), ua_to_bv([0, 0, 1]))]), dict_decompress);
                    assert_eq!(array, array_compressed_decompressed);
                }
                #[test]
                fn elements_false_true_true() {
                    const N: usize = 12;
                    const WORD_LEN: usize = 3;
                    let array: Array2dBool<N> = Array2dBool::from_array(ua_to_ba([0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1]));
                    let huffman_compressed: HuffmanCompressed = array.clone().compress_by_huffman(WORD_LEN);
                    let array_compressed_decompressed: Array2dBool<N> = huffman_compressed.decompress();
                    let (vec_compressed, dict_decompress) = huffman_compressed.get_vec_and_dict();
                    assert_eq!(ua_to_bv([0, 0, 0, 0]), vec_compressed);
                    assert_eq!(HashMap::from([(ua_to_bv([0]), ua_to_bv([0, 1, 1]))]), dict_decompress);
                    assert_eq!(array, array_compressed_decompressed);
                }
                #[test]
                fn elements_true_true_true() {
                    const N: usize = 12;
                    const WORD_LEN: usize = 3;
                    let array: Array2dBool<N> = Array2dBool::from_array(ua_to_ba([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]));
                    let huffman_compressed: HuffmanCompressed = array.clone().compress_by_huffman(WORD_LEN);
                    let array_compressed_decompressed: Array2dBool<N> = huffman_compressed.decompress();
                    let (vec_compressed, dict_decompress) = huffman_compressed.get_vec_and_dict();
                    assert_eq!(ua_to_bv([0, 0, 0, 0]), vec_compressed);
                    assert_eq!(HashMap::from([(ua_to_bv([0]), ua_to_bv([1, 1, 1]))]), dict_decompress);
                    assert_eq!(array, array_compressed_decompressed);
                }
                #[test]
                fn elements_different() {
                    const N: usize = 12;
                    const WORD_LEN: usize = 3;
                    let array: Array2dBool<N> = Array2dBool::from_array(ua_to_ba([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]));
                    let huffman_compressed: HuffmanCompressed = array.clone().compress_by_huffman(WORD_LEN);
                    let array_compressed_decompressed: Array2dBool<N> = huffman_compressed.decompress();
                    let (vec_compressed, dict_decompress) = huffman_compressed.get_vec_and_dict();
                    assert!(
                        ua_to_bv([0, 1, 0, 1]) == vec_compressed ||
                        ua_to_bv([1, 0, 1, 0]) == vec_compressed
                    );
                    assert!(
                        HashMap::from([(ua_to_bv([0]), ua_to_bv([0, 0, 0])), (ua_to_bv([1]), ua_to_bv([1, 1, 1]))]) == dict_decompress ||
                        HashMap::from([(ua_to_bv([1]), ua_to_bv([0, 0, 0])), (ua_to_bv([0]), ua_to_bv([1, 1, 1]))]) == dict_decompress
                    );
                    assert_eq!(array, array_compressed_decompressed);
                }
            }
        }

        mod array_size_100 {
            mod word_len_5 {
                use super::super::super::super::*;
                use crate::utils::ua_to_ba;
                #[test]
                fn elements_random() {
                    const N: usize = 100;
                    const WORD_LEN: usize = 5;
                    let array: Array2dBool<N> = Array2dBool::from_array(ua_to_ba([
                        0, 1, 0, 0, 1, 0, 0, 0, 1, 0,
                        0, 1, 0, 0, 1, 1, 1, 1, 1, 1,
                        0, 1, 1, 0, 1, 0, 1, 0, 0, 0,
                        1, 0, 0, 1, 1, 0, 1, 0, 0, 1,
                        0, 1, 1, 0, 1, 1, 0, 0, 1, 1,
                        0, 1, 0, 1, 0, 0, 0, 1, 1, 0,
                        0, 1, 1, 0, 1, 1, 1, 1, 1, 0,
                        0, 1, 1, 1, 0, 0, 0, 1, 0, 0,
                        1, 0, 0, 1, 1, 0, 0, 0, 1, 1,
                        0, 0, 1, 0, 0, 1, 1, 1, 0, 0,
                    ]));
                    let huffman_compressed: HuffmanCompressed = array.clone().compress_by_huffman(WORD_LEN);
                    let array_compressed_decompressed: Array2dBool<N> = huffman_compressed.decompress();
                    println!("calc_weight = {weight}", weight = huffman_compressed.calc_weight());
                    // assert_eq!(bafuv([]), vec_compressed);
                    // assert_eq!(HashMap::from([]), dict_decompress);
                    assert_eq!(array, array_compressed_decompressed);
                }

            }
        }

    }

}

