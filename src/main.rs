//! Bad Apple!!
//!
//! crop video to frames command:
//! `ffmpeg -i bad_apple.mp4 -qscale:v 1 f%04d.png`

// #![feature(adt_const_params)]
// #![feature(generic_const_exprs)]



use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::io::Write;
use std::ops::{Index, IndexMut};
use std::thread::sleep;
use std::time::{Duration, Instant};

use image::io::Reader as ImageReader;
use image::{Rgb, RgbImage};



const FRAME_H: usize = 20;
const FRAME_W: usize = FRAME_H * FRAME_W_MAX / FRAME_H_MAX;

const FRAME_W_MAX: usize = 480;
const FRAME_H_MAX: usize = 360;

const FRAME_WH: usize = FRAME_W * FRAME_H;

const FRAMES_AMOUNT: usize = 6572;

// const PIXELS_AMOUNT: usize = FRAME_WH * FRAMES_AMOUNT;



pub trait ExtensionTo01String {
    fn to_01_string(&self) -> String;
}
impl ExtensionTo01String for Vec<bool> {
    fn to_01_string(&self) -> String {
        self.iter().fold("".to_string(), |acc, &it| acc + if it == false { "0" } else { "1" })
    }
}

pub trait ExtensionToArray<T, const N: usize> {
    fn to_array(self) -> [T; N];
}
impl<T, const N: usize> ExtensionToArray<T, N> for Vec<T> {
    fn to_array(self) -> [T; N] {
        self.try_into()
            .unwrap_or_else(|vec: Vec<T>|
                panic!("Expected a Vec of length {} but it was {}", N, vec.len())
            )
    }
}



pub trait HashMapExtensionSetOneOrIncreaseByOne<K> {
    fn set_or_inc(&mut self, key: K);
}
impl<K: Eq+Hash+Clone> HashMapExtensionSetOneOrIncreaseByOne<K> for HashMap<K, u32> {
    fn set_or_inc(&mut self, key: K) {
        if !self.contains_key(&key) {
            self.insert(key, 1);
        }
        else {
            self.insert(key.clone(), self[&key]+1);
        }
    }
}

pub trait HashMapExtensionGetKeyWithBiggestValue<K> {
    fn get_key_with_biggest_value(&self) -> &K;
}
impl<K: Eq+Hash> HashMapExtensionGetKeyWithBiggestValue<K> for HashMap<K, u32> {
    fn get_key_with_biggest_value(&self) -> &K {
        let mut best_key: &K = self.keys().next().unwrap();
        let mut best_value: u32 = self[&best_key];
        for (key, value) in self.iter() {
            if *value > best_value {
                best_key = key;
                best_value = *value;
            }
        }
        &best_key
    }
}

// TODO: rewrote without clone?
pub trait HashMapExtensionInvert<K, V> {
    fn invert(&self) -> HashMap<V, K>;
}
impl<K, V> HashMapExtensionInvert<K, V> for HashMap<K, V>
where
    K: Clone,
    V: Clone + Eq + Hash
{
    fn invert(&self) -> HashMap<V, K> {
        self.iter()
            .fold(HashMap::new(), |mut acc, (k, v)| {
                acc.insert(v.clone(), k.clone());
                acc
            })
    }
}



/// immutably adds vectors
pub trait VecExtensionAdd<T> {
    fn add(&self, other: Vec<T>) -> Vec<T>;
}
impl<T: Clone> VecExtensionAdd<T> for Vec<T> {
    fn add(&self, other: Vec<T>) -> Vec<T> {
        let len: usize = self.len() + other.len();
        let mut res: Vec<T> = Vec::with_capacity(len);
        res.extend(self.clone());
        res.extend(other);
        res
    }
}

pub trait VecExtensionSorted<T> {
    fn sorted(&self) -> Vec<T>;
}
impl<T: Clone + Ord> VecExtensionSorted<T> for Vec<T> {
    fn sorted(&self) -> Vec<T> {
        let mut res: Vec<T> = self.clone();
        res.sort();
        res
    }
}



// pub trait U64ExtensionDivWithNDigitsAfterDecimalPoint {
//     fn div_with_n_digits_after_decimal_point(&self, other: u64, n_digits_after_decimal_point: u8) -> f64;
// }
// impl U64ExtensionDivWithNDigitsAfterDecimalPoint for u64 {
//     fn div_with_n_digits_after_decimal_point(&self, other: u64, n: u8) -> f64 {
//         let sh: u64 = 10_u32.pow(n as u32) as u64;
//         (sh * self / other) as f64 / (sh as f64)
//     }
// }



pub trait Len {
    fn len(&self) -> usize;
}
impl<T, const N: usize> Len for [T; N] {
    fn len(&self) -> usize { <[T]>::len(self) }
}
impl<T> Len for Vec<T> {
    fn len(&self) -> usize { Vec::<T>::len(self) }
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



type WordUncompressed = Vec<bool>;
type WordCompressed   = Vec<bool>;

type VecCompressed  = Vec<bool>;
type DictCompress   = HashMap<WordUncompressed, WordCompressed>;
type DictDecompress = HashMap<WordCompressed, WordUncompressed>;

type Frame = Array2dBool<FRAME_WH>;



// /// bool means:
// /// - true  -> white
// /// - false -> black
// #[derive(Copy, Clone, Debug, PartialEq)]
// pub struct Array2dBool<const N: usize> {
//     elements: [bool; N]
// }

// impl<const N: usize> Array2dBool<N> {
//     pub const fn new(fill_by: bool) -> Array2dBool<N> {
//         Array2dBool { elements: [fill_by; N] }
//     }

//     pub const fn from_array(elements: [bool; N]) -> Array2dBool<N> {
//         Array2dBool { elements }
//     }

//     pub const fn calc_weight(&self) -> usize { N }

//     /// returns tuple of
//     /// - compressed array
//     /// - dict for decompression
//     pub fn compress(&self, word_len: usize) -> HuffmanCompressed {
//         compress_any_array(self.elements, word_len)
//     }

// }

// // pub const fn wh_to_index<const W: usize>(w: usize, h: usize) -> usize { w + h * W }
// // pub const fn index_to_wh<const W: usize>(index: usize) -> (usize, usize) { (index % W, index / W) }

// impl<const N: usize> Index<usize> for Array2dBool<N> {
//     type Output = bool;
//     fn index(&self, index: usize) -> &Self::Output {
//         &self.elements[index]
//     }
// }

// impl<const N: usize> IndexMut<usize> for Array2dBool<N> {
//     fn index_mut(&mut self, index: usize) -> &mut Self::Output {
//         &mut self.elements[index]
//     }
// }





/// bool means:
/// - true  -> white
/// - false -> black
#[derive(Clone, Debug, PartialEq)]
pub struct Array2dBool<const N: usize> {
    elements: Vec<bool>
}

impl<const N: usize> Array2dBool<N> {
    pub fn new(fill_by: bool) -> Array2dBool<N> {
        Array2dBool { elements: vec![fill_by; N] }
    }

    pub fn from_vec(elements: Vec<bool>) -> Array2dBool<N> {
        Array2dBool { elements }
    }

    pub fn from_array(elements: [bool; N]) -> Array2dBool<N> {
        Array2dBool { elements: elements.to_vec() }
    }

    pub const fn calc_weight(&self) -> usize { N }

    /// returns tuple of
    /// - compressed array
    /// - dict for decompression
    pub fn compress_by_huffman(&self, word_len: usize) -> HuffmanCompressed {
        HuffmanCompressed::compress_from(&self.elements, word_len)
    }

    /// TODO:
    pub fn compress_by_sequence(&self) -> SequenceCompressed {
        SequenceCompressed::compress_from(&self.elements)
    }

}

pub const fn wh_to_index<const W: usize>(w: usize, h: usize) -> usize { w + h * W }
pub const fn index_to_wh<const W: usize>(index: usize) -> (usize, usize) { (index % W, index / W) }

impl<const N: usize> Index<usize> for Array2dBool<N> {
    type Output = bool;
    fn index(&self, index: usize) -> &Self::Output {
        &self.elements[index]
    }
}

impl<const N: usize> IndexMut<usize> for Array2dBool<N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.elements[index]
    }
}

// impl<const N: usize> Index<(usize, usize)> for Array2dBool<N> {
//     type Output = bool;
//     fn index(&self, (w, h): (usize, usize)) -> &Self::Output {
//         &self.elements[wh_to_index(w, h)]
//     }
// }

// impl<const N: usize> IndexMut<(usize, usize)> for Array2dBool<N> {
//     fn index_mut(&mut self, (w, h): (usize, usize)) -> &mut Self::Output {
//         &mut self.elements[wh_to_index(w, h)]
//     }
// }



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
        self.vec_compressed.len() as u64 +
            self.dict_decompress.iter()
                .map(|(key, value)| key.len() as u64 + value.len() as u64 + 1)
                .sum::<u64>()
    }

}



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
        // TODO
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

    /// get VecCompressed
    pub fn get_vec(self) -> Sequence { self.vec_compressed }

    pub fn calc_weight(&self) -> u64 {
        self.vec_compressed.iter()
            .map(|&it| it as u64)
            // TODO: map 42 = 101010 -> 6
            .map(|it| format!("{:b}", it).len() as u64 + 1)
            .sum()
    }

}





// const PATH_TO_IMAGE: &str = "../data/frames/f240.png";
const COLOR_THRESHOLD: u8 = 127;

pub fn path_to_nth_input_frame(n: usize) -> String {
    match n {
        n if (1..10).contains(&n) => { format!("../data/frames/f000{}.png", n) }
        n if (10..100).contains(&n) => { format!("../data/frames/f00{}.png", n) }
        n if (100..1000).contains(&n) => { format!("../data/frames/f0{}.png", n) }
        n if (1000..=FRAMES_AMOUNT).contains(&n) => { format!("../data/frames/f{}.png", n) }
        _ => { panic!("path_to_nth_input_frame: Bad `n`: {}", n) }
    }
}

pub fn load_frames() -> Vec<Frame> {
    (0..FRAMES_AMOUNT).map(load_frame).collect()
}

pub fn load_frame<const N: usize>(i: usize) -> Array2dBool<N> {
    // println!("loading frame {i} / {FRAMES_AMOUNT}");
    let img = ImageReader::open(path_to_nth_input_frame(i+1)).unwrap().decode().unwrap();

    // assert image sizes
    {
        let (image_w, image_h) = img.as_rgb8().unwrap().dimensions();
        let (image_w, image_h) = (image_w as usize, image_h as usize);
        assert_eq!(FRAME_W_MAX, image_w);
        assert_eq!(FRAME_H_MAX, image_h);
    }

    // create pixels_rgb array
    let mut pixels_rgb: Vec<Vec<Rgb<u8>>> = Vec::with_capacity(FRAME_W);
    for _ in 0..FRAME_W {
        pixels_rgb.push(Vec::with_capacity(FRAME_H));
    }
    // load pixels_rgb
    let img_rgb: &RgbImage = img.as_rgb8().unwrap();
    for w in 0..FRAME_W {
        for h in 0..FRAME_H {
            pixels_rgb[w].push(*img_rgb.get_pixel(
                (w * FRAME_W_MAX / FRAME_W) as u32,
                (h * FRAME_H_MAX / FRAME_H) as u32
            ));
        }
    }
    // assert pixels_rgb
    assert_eq!(FRAME_W, pixels_rgb.len());
    assert!(pixels_rgb.iter().all(|it| it.len() == FRAME_H));
    assert_eq!(FRAME_WH, pixels_rgb.iter().fold(0, |acc, el| acc + el.len()));

    let pixels_bool_2d: Vec<Vec<bool>> = pixels_rgb.into_iter().map(|col: Vec<Rgb<u8>>|
        col.iter().map(|it: &Rgb<u8>|
            (it[0] > COLOR_THRESHOLD || it[1] > COLOR_THRESHOLD || it[2] > COLOR_THRESHOLD)
        ).collect()
    ).collect();

    let pixels_bool: Vec<bool> = {
        let mut pixels_bool: Vec<bool> = Vec::with_capacity(FRAME_WH);
        for h in 0..FRAME_H {
            for w in 0..FRAME_W {
                pixels_bool.push(pixels_bool_2d[w][h]);
            }
        }
        pixels_bool
    };
    // assert pixels_bool
    assert_eq!(FRAME_WH, pixels_bool.len());

    let array2d: Array2dBool<N> = Array2dBool::from_array(pixels_bool.to_array());

    // pixels_bool
    array2d
}

impl ToString for Frame {
    fn to_string(&self) -> String {
        const STRING_CAPACITY: usize = FRAME_WH*2 + FRAME_H + 1;
        let mut res: String = String::with_capacity(STRING_CAPACITY);
        for i in 0..FRAME_WH {
            if i != 0 && i % FRAME_W == 0 { res += "\n"; }
            // res += if self[i] { "1 " } else { "0 " };
            // res += if self[i] { "██" } else { "  " };
            res += if self[i] { "@@" } else { "  " };
        }
        res
    }
}

pub fn show_video() {
    // all times here in seconds
    const RENDER_TIME: f64 = 0.0 * 60.0 + 48.67;
    const VIDEO_TIME : f64 = 3.0 * 60.0 + 39.00;
    const FRAME_DELAY_TIME: f64 = (VIDEO_TIME - RENDER_TIME) / FRAMES_AMOUNT as f64;
    // const FRAME_DELAY_TIME: f64 = 0.1;

    const FPS_DECREASE_K: usize = 1;

    print!("\nLoading frames... "); flush();
    let mut frames: Vec<Frame> = Vec::with_capacity(FRAMES_AMOUNT);
    let time = measure_time(|| {
        frames = (0..FRAMES_AMOUNT).map(load_frame).collect();
    });
    println!("Loaded in {time:.2} s\n");

    // this is needed for "clearing" screen
    let new_lines: String = "\n".repeat(100);

    let time = measure_time(|| {
        for i in (0..FRAMES_AMOUNT).step_by(FPS_DECREASE_K) {
            let frame = &frames[i];
            print!("{new_lines}{frame}", frame = frame.to_string()); flush();
            // TODO: try exactly clearing screen, not printing \n's
            sleep(Duration::from_micros((FPS_DECREASE_K as f64 * 1_000_000.0 * FRAME_DELAY_TIME) as u64));
        }
    });
    println!("Rendered video in {time:.2} s");
}



/// diffrent types of nested cycles of FRAME_W, FRAME_H, FRAMES_AMOUNT (N)
#[derive(Copy, Clone, Debug)]
pub enum FramesOrganisation {
    HNW, HWN, NHW, NWH, WHN, WNH
}

pub fn frames_to_array2d(frames: Vec<Frame>, frames_organisation: FramesOrganisation) -> Array2dBool<{FRAME_WH*FRAMES_AMOUNT}> {
    const W: usize = FRAME_W;
    const H: usize = FRAME_H;
    const N: usize = FRAMES_AMOUNT;
    let mut pixels_all: Array2dBool<{FRAME_WH*FRAMES_AMOUNT}> = Array2dBool::new(false);
    match frames_organisation {
        FramesOrganisation::HNW => {
            for h in 0..H {
                for w in 0..W {
                    for n in 0..N {
                        pixels_all[h*W*N + n*W + w] = frames[n][w + h*W];
                    }
                }
            }
        }
        FramesOrganisation::HWN => {
            for h in 0..H {
                for w in 0..W {
                    for n in 0..N {
                        pixels_all[h*W*N + w*N + n] = frames[n][w + h*W];
                    }
                }
            }
        }
        FramesOrganisation::NHW => {
            for h in 0..H {
                for w in 0..W {
                    for n in 0..N {
                        pixels_all[n*W*H + h*W + w] = frames[n][w + h*W];
                    }
                }
            }
        }
        FramesOrganisation::NWH => {
            for h in 0..H {
                for w in 0..W {
                    for n in 0..N {
                        pixels_all[n*W*H + w*H + h] = frames[n][w + h*W];
                    }
                }
            }
        }
        FramesOrganisation::WHN => {
            for h in 0..H {
                for w in 0..W {
                    for n in 0..N {
                        pixels_all[w*H*N + h*N + n] = frames[n][w + h*W];
                    }
                }
            }
        }
        FramesOrganisation::WNH => {
            for h in 0..H {
                for w in 0..W {
                    for n in 0..N {
                        pixels_all[w*H*N + n*H + h] = frames[n][w + h*W];
                    }
                }
            }
        }
    }
    pixels_all
}



pub fn compress_video_by_huffman_with_word_len(pixels: &Array2dBool<{FRAME_WH*FRAMES_AMOUNT}>, word_len: usize) -> HuffmanCompressed {
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



fn main() {
    // println!("FRAMES_AMOUNT = {FRAMES_AMOUNT}");

    // show_video();

    // compress_video_by_huffman();
    compress_by_sequence();

    // let mut x: i32;
    // let mut f = || {
    //     x = 1;
    // };
    // f();
    // println!("x = {x}");

    println!("\nProgram finished successfully!");
}





#[cfg(test)]
mod tests {

    fn ua_to_ba<const N: usize>(array_u32: [u32; N]) -> [bool; N] {
        array_u32.map(|it| match it {
            0 => { false }
            1 => { true }
            _ => { panic!() }
        })
    }

    fn ua_to_bv<const N: usize>(array_u32: [u32; N]) -> Vec<bool> {
        array_u32.map(|it| match it {
            0 => { false }
            1 => { true }
            _ => { panic!() }
        }).to_vec()
    }

    mod build_best_dictionary {
        use std::collections::HashMap;
        use crate::{build_best_dictionary, WordCompressed};
        use crate::tests::ua_to_bv;

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
                use crate::{Array2dBool, HuffmanCompressed};
                use crate::tests::ua_to_bv;
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
                use crate::{Array2dBool, HuffmanCompressed};
                use crate::tests::{ua_to_ba, ua_to_bv};
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
                use crate::{Array2dBool, HuffmanCompressed};
                use crate::tests::{ua_to_ba, ua_to_bv};
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
                use crate::{Array2dBool, HuffmanCompressed};
                use crate::tests::ua_to_ba;
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



    mod compress_decompress_by_sequence {
        use crate::{Array2dBool, SequenceCompressed};
        use crate::tests::ua_to_ba;
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

