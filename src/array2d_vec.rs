//! Array2d impl by dynamic array

use std::ops::{
    Index,
    IndexMut,
};

use crate::compressions::{
    huffman::HuffmanCompressed,
    sequence::SequenceCompressed,
};



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

    /// returns
    /// - compressed array
    /// - dict for decompression
    pub fn compress_by_huffman(&self, word_len: usize) -> HuffmanCompressed {
        HuffmanCompressed::compress_from(&self.elements, word_len)
    }

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

