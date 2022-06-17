//! Bad Apple!!
//!
//! crop video to frames command:
//! `ffmpeg -i bad_apple.mp4 -qscale:v 1 f%04d.png`

// #![feature(adt_const_params)]
// #![feature(generic_const_exprs)]



use std::fmt::Debug;
use std::thread::sleep;
use std::time::Duration;

use image::io::Reader as ImageReader;
use image::{Rgb, RgbImage};

// mod array2d;
mod array2d_vec;
mod compressions;
mod extensions;
mod type_aliases;
mod utils;

use crate::{
    array2d_vec::*,
    compressions::{
        huffman::HuffmanCompressed,
        sequence::SequenceCompressed,
    },
    extensions::*,
    type_aliases::Frame,
    utils::*,
};



pub const FRAME_H: usize = 20;
pub const FRAME_W: usize = FRAME_H * FRAME_W_MAX / FRAME_H_MAX;

pub const FRAME_W_MAX: usize = 480;
pub const FRAME_H_MAX: usize = 360;

pub const FRAME_WH: usize = FRAME_W * FRAME_H;

pub const FRAMES_AMOUNT: usize = 6572;

// const PIXELS_AMOUNT: usize = FRAME_WH * FRAMES_AMOUNT;





fn main() {
    // println!("FRAMES_AMOUNT = {FRAMES_AMOUNT}");

    show_video();

    // compress_video_by_huffman();
    // compress_by_sequence();

    // let mut x: i32;
    // let mut f = || {
    //     x = 1;
    // };
    // f();
    // println!("x = {x}");

    println!("\nProgram finished successfully!");
}







// const PATH_TO_IMAGE: &str = "../data/frames/f240.png";
pub const COLOR_THRESHOLD: u8 = 127;

pub fn path_to_nth_input_frame(n: usize) -> String {
    match n {
        n if (1..10).contains(&n) => { format!("data/frames/f000{}.png", n) }
        n if (10..100).contains(&n) => { format!("data/frames/f00{}.png", n) }
        n if (100..1000).contains(&n) => { format!("data/frames/f0{}.png", n) }
        n if (1000..=FRAMES_AMOUNT).contains(&n) => { format!("data/frames/f{}.png", n) }
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







#[cfg(test)]
mod tests {

    mod build_best_dictionary {
        use std::collections::HashMap;
        use crate::compressions::huffman::{build_best_dictionary, WordCompressed};
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
                use crate::{Array2dBool, HuffmanCompressed};
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
                use crate::{Array2dBool, HuffmanCompressed};
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
                use crate::{Array2dBool, HuffmanCompressed};
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
                use crate::{Array2dBool, HuffmanCompressed};
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



    mod compress_decompress_by_sequence {
        use crate::{Array2dBool, SequenceCompressed};
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

