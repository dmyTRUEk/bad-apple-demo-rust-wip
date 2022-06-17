//! Frame related stuff

use std::fmt::Debug;
use std::thread::sleep;
use std::time::Duration;

use image::io::Reader as ImageReader;
use image::{Rgb, RgbImage};

use crate::{
    array2d_vec::Array2dBool,
    extensions::ExtensionToArray,
    utils::{flush, measure_time},
};



pub const FRAME_H: usize = 20;
pub const FRAME_W: usize = FRAME_H * FRAME_W_MAX / FRAME_H_MAX;

pub const FRAME_W_MAX: usize = 480;
pub const FRAME_H_MAX: usize = 360;

pub const FRAME_WH: usize = FRAME_W * FRAME_H;

pub const FRAMES_AMOUNT: usize = 6572;

// const PIXELS_AMOUNT: usize = FRAME_WH * FRAMES_AMOUNT;



pub type Frame = Array2dBool<FRAME_WH>;



pub fn path_to_nth_input_frame(n: usize) -> String {
    match n {
        n if (1..=FRAMES_AMOUNT).contains(&n) => {
            format!("data/frames/f{n:0>4.}.png")
        }
        _ => { panic!("path_to_nth_input_frame: Bad `n`: {}", n) }
    }
}

pub fn load_frames() -> Vec<Frame> {
    (0..FRAMES_AMOUNT).map(load_frame).collect()
}

pub const COLOR_THRESHOLD: u8 = 127;
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
    assert_eq!(FRAME_W, pixels_rgb.len());
    assert!(pixels_rgb.iter().all(|it| it.len() == FRAME_H));
    assert_eq!(FRAME_WH, pixels_rgb.iter().fold(0, |acc, el| acc + el.len()));

    let pixels_bool_2d: Vec<Vec<bool>> = pixels_rgb.into_iter().map(|col: Vec<Rgb<u8>>|
        col.iter().map(|it: &Rgb<u8>|
            (it[0] > COLOR_THRESHOLD || it[1] > COLOR_THRESHOLD || it[2] > COLOR_THRESHOLD)
        ).collect()
    ).collect();

    let mut pixels_bool: Vec<bool> = Vec::with_capacity(FRAME_WH);
    for h in 0..FRAME_H {
        for w in 0..FRAME_W {
            pixels_bool.push(pixels_bool_2d[w][h]);
        }
    }
    let pixels_bool: Vec<bool> = pixels_bool;
    assert_eq!(FRAME_WH, pixels_bool.len());

    let array2d: Array2dBool<N> = Array2dBool::from_array(pixels_bool.to_array());

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

pub fn frames_to_array2d(
    frames: Vec<Frame>,
    frames_organisation: FramesOrganisation
) -> Array2dBool<{FRAME_WH*FRAMES_AMOUNT}> {
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

