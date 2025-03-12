#![allow(unused)]
use std::{
    hash::{Hash, Hasher},
    iter::Iterator,
    num::NonZeroU64,
};

#[derive(PartialEq, Clone, Debug)]
struct Bloom {
    buffer: Box<[u8]>,
    num_slices: NonZeroU64,
}

impl Bloom {
    fn new(capacity: usize, error_ratio: f64) -> Bloom {
        // Directly from paper:
        // k = log2(1/P)   (num_slices)
        // n ≈ −m ln(1−p)  (slice_len_bits)
        // M = k * m       (total_bits)
        // for optimal filter p = 0.5, which gives:
        // n ≈ −m ln(0.5), rearranging: m = -n / ln(0.5) = n / ln(2)
        /* debug_assert!(capacity >= 1);
        debug_assert!(0.0 < error_ratio && error_ratio < 1.0); */
        // We're using ceil instead of round in order to get an error rate <= the desired.
        // Using round can result in significantly higher error rates.
        let num_slices = ((1.0 / error_ratio).log2()).ceil() as u64;
        let slice_len_bits = (capacity as f64 / 2f64.ln()).ceil() as u64;
        let total_bits = num_slices * slice_len_bits;
        // round up to the next byte
        let buffer_bytes = ((total_bits + 7) / 8) as usize;

        let buffer = vec![0; buffer_bytes];
        Bloom {
            buffer: buffer.into_boxed_slice(),
            num_slices: NonZeroU64::new(num_slices).unwrap(),
        }
    }

    #[inline]
    fn index_iterator(&self, mut h1: u64, mut h2: u64) -> impl Iterator<Item = (usize, u8)> {
        // The _bit_ length (thus buffer.len() multiplied by 8) of each slice within buffer.
        // We'll use a NonZero type so that the compiler can avoid checking for
        // division/modulus by 0 inside the iterator.
        let slice_len = NonZeroU64::new(self.buffer.len() as u64 * 8 / self.num_slices).unwrap();

        // Generate `self.num_slices` hashes from 2 hashes, using enhanced double hashing.
        // See https://en.wikipedia.org/wiki/Double_hashing#Enhanced_double_hashing for details.
        // We choose to use 2x64 bit hashes instead of 2x32 ones as it gives significant better false positive ratios.
        // debug_assert_ne!(h2, 0, "Second hash can't be 0 for double hashing");
        (0..self.num_slices.get()).map(move |i| {
            // Calculate hash(i)
            let hi = h1 % slice_len + i * slice_len.get();
            // Advance enhanced double hashing state
            h1 = h1.wrapping_add(h2);
            h2 = h2.wrapping_add(i);
            // Resulting index/mask based on hash(i)
            let idx = (hi / 8) as usize;
            let mask = 1u8 << (hi % 8);
            (idx, mask)
        })
    }

    #[inline]
    fn insert(&mut self, h1: u64, h2: u64) {
        // Set all bits (one per slice) corresponding to this item.
        //
        // Setting the bit:
        //    1000 0011 (self.buffer[idx])
        //    0001 0000 (mask)
        //    |---------
        //    1001 0011
        //
        for (byte, mask) in self.index_iterator(h1, h2) {
            self.buffer[byte] |= mask;
        }
    }

    #[inline]
    fn contains(&self, h1: u64, h2: u64) -> bool {
        // Check if all bits (one per slice) corresponding to this item are set.
        // See index_iterator comments for a detailed explanation.
        //
        // Potentially found case:
        //    0111 1111 (self.buffer[idx])
        //    0001 0000 (mask)
        //    &---------
        //    0001 0000 != 0
        //
        // Definitely not found case:
        //    1110 1111 (self.buffer[idx])
        //    0001 0000 (mask)
        //    &---------
        //    0000 0000 == 0
        //
        self.index_iterator(h1, h2)
            .all(|(byte, mask)| self.buffer[byte] & mask != 0)
    }
}

#[inline]
pub fn double_hashing_hashes<T, H>(item: T, mut hasher: H) -> (u64, u64)
where
    T: Hash,
    H: Hasher,
{
    item.hash(&mut hasher);
    let h1 = hasher.finish();

    // Write a nul byte to the existing state and get another hash.
    // This is appropriate when using a very high quality hasher,
    // which we know is the case.
    0u8.hash(&mut hasher);
    // h2 hash shouldn't be 0 for double hashing
    let h2 = hasher.finish().max(1);

    (h1, h2)
}

// From the paper:
// Considering the choice of s (GROWTH_FACTOR) = 2 for small expected growth and s = 4
// for larger growth, one can see that r (TIGHTENING_RATIO) around 0.8 – 0.9 is a sensible choice.
// Here we select good defaults for 10~1000x growth.
const DEFAULT_GROWTH_FACTOR: usize = 2;
const DEFAULT_TIGHTENING_RATIO: f64 = 0.8515625; // ~0.85 but has exact representation in f32/f64

#[derive(PartialEq, Clone, Debug)]
pub struct BloomFilter {
    blooms: Vec<Bloom>,
    desired_error_prob: f64,
    est_insertions: usize,

    inserts: usize,

    capacity: usize,

    growth_factor: usize,
    tightening_ratio: f64,
}

impl BloomFilter {
    #[inline]
    pub fn new(desired_error_prob: f64, est_insertions: usize) -> BloomFilter {
        Self::new_with_internals(
            desired_error_prob,
            est_insertions,
            DEFAULT_GROWTH_FACTOR,
            DEFAULT_TIGHTENING_RATIO,
        )
    }

    pub(crate) fn new_with_internals(
        desired_error_prob: f64,
        est_insertions: usize,
        growth_factor: usize,
        tightening_ratio: f64,
    ) -> BloomFilter {
        assert!(0.0 < desired_error_prob && desired_error_prob < 1.0);
        assert!(growth_factor > 1);
        BloomFilter {
            blooms: vec![],
            desired_error_prob,
            est_insertions,
            inserts: 0,
            capacity: 0,
            growth_factor,
            tightening_ratio,
        }
    }

    pub fn contains<T: Hash, H: Hasher>(&self, item: T, hasher: H) -> bool {
        let (h1, h2) = double_hashing_hashes(item, hasher);
        self.blooms.iter().any(|bloom| bloom.contains(h1, h2))
    }

    pub fn contains_raw(&self, (h1, h2): (u64, u64)) -> bool {
        self.blooms.iter().any(|bloom| bloom.contains(h1, h2))
    }

    pub fn insert<T: Hash, H: Hasher>(&mut self, item: T, hasher: H) -> bool {
        let (h1, h2) = double_hashing_hashes(item, hasher);
        // Step 1: Ask if we already have it
        if self.blooms.iter().any(|bloom| bloom.contains(h1, h2)) {
            return false;
        }
        // Step 2: Grow if necessary
        if self.inserts >= self.capacity {
            self.grow();
        }
        // Step 3: Insert it into the last
        self.inserts += 1;
        let curr_bloom = self.blooms.last_mut().unwrap();
        curr_bloom.insert(h1, h2);
        true
    }

    pub fn insert_raw(&mut self, (h1, h2): (u64, u64)) -> bool {
        // Step 1: Ask if we already have it
        if self.blooms.iter().any(|bloom| bloom.contains(h1, h2)) {
            return false;
        }
        // Step 2: Grow if necessary
        if self.inserts >= self.capacity {
            self.grow();
        }
        // Step 3: Insert it into the last
        self.inserts += 1;
        let curr_bloom = self.blooms.last_mut().unwrap();
        curr_bloom.insert(h1, h2);
        true
    }

    pub fn clear(&mut self) {
        self.blooms.clear();
        self.inserts = 0;
        self.capacity = 0;
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inserts == 0
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.inserts
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn check_and_set<T: Hash, H: Hasher>(&mut self, item: T, hasher: H) -> bool {
        !self.insert(item, hasher)
    }

    fn grow(&mut self) {
        // The paper gives an upper bound formula for the fp rate: fpUB <= fp0 * / (1-r)
        // This is because each sub bloom filter is created with an ever smaller
        // false-positive ratio, forming a geometric progression.
        // let r = TIGHTENING_RATIO
        // fpUB ~= fp0 * fp0*r * fp0*r*r * fp0*r*r*r ...
        // fp(x) = fp0 * (r**x)
        let error_ratio =
            self.desired_error_prob * self.tightening_ratio.powi(self.blooms.len() as _);
        // In order to have relatively small space overhead compared to a single appropriately sized bloom filter
        // the sub filters should be created with increasingly bigger sizes.
        // let s = GROWTH_FACTOR
        // cap(x) = cap0 * (s**x)
        let capacity = self.est_insertions * self.growth_factor.pow(self.blooms.len() as _);
        let new_bloom = Bloom::new(capacity, error_ratio);
        self.blooms.push(new_bloom);
        self.capacity += capacity;
    }
}
