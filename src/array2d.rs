//! Array2d impl by static array



/// bool means:
/// - true  -> white
/// - false -> black
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Array2dBool<const N: usize> {
    elements: [bool; N]
}

impl<const N: usize> Array2dBool<N> {
    pub const fn new(fill_by: bool) -> Array2dBool<N> {
        Array2dBool { elements: [fill_by; N] }
    }

    pub const fn from_array(elements: [bool; N]) -> Array2dBool<N> {
        Array2dBool { elements }
    }

    pub const fn calc_weight(&self) -> usize { N }

    /// returns tuple of
    /// - compressed array
    /// - dict for decompression
    pub fn compress(&self, word_len: usize) -> HuffmanCompressed {
        compress_any_array(self.elements, word_len)
    }

}

// pub const fn wh_to_index<const W: usize>(w: usize, h: usize) -> usize { w + h * W }
// pub const fn index_to_wh<const W: usize>(index: usize) -> (usize, usize) { (index % W, index / W) }

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

