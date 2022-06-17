//! Huffman compression

use std::{
    collections::HashMap,
    fmt::Debug,
    hash::Hash,
    ops::Index,
};

use crate::{
    Array2dBool,
    extensions::{HashMapExtensionInvert, HashMapExtensionSetOneOrIncreaseByOne, VecExtensionAdd},
    Len,
};



pub type WordUncompressed = Vec<bool>;
pub type WordCompressed   = Vec<bool>;

pub type VecCompressed  = Vec<bool>;
pub type DictCompress   = HashMap<WordUncompressed, WordCompressed>;
pub type DictDecompress = HashMap<WordCompressed, WordUncompressed>;



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

