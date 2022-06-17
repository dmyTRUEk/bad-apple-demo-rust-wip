// Extensions

use std::{
    collections::HashMap,
    hash::Hash,
};



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



pub trait Len {
    fn len(&self) -> usize;
}
impl<T, const N: usize> Len for [T; N] {
    fn len(&self) -> usize { <[T]>::len(self) }
}
impl<T> Len for Vec<T> {
    fn len(&self) -> usize { Vec::<T>::len(self) }
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

