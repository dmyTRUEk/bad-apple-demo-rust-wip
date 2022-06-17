//! Bad Apple!!
//!
//! crop video to frames command:
//! `ffmpeg -i bad_apple.mp4 -qscale:v 1 f%04d.png`

//#![feature(adt_const_params)]
//#![feature(generic_const_exprs)]



//mod array2d;
mod array2d_vec;
mod compressions;
mod extensions;
mod frame;
mod utils;

use crate::frame::show_video;



fn main() {
    show_video();

    // compress_video_by_huffman();
    // compress_by_sequence();

    println!("\nProgram finished successfully!");
}

