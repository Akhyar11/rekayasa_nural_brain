use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RandomProjectionVector {
    pub values: Vec<f32>,
}

impl RandomProjectionVector {
    pub const DIM: usize = 64;

    pub fn for_token(token_id: u64) -> Self {
        let mut state = token_id ^ 0x9e3779b97f4a7c15;
        let mut values = Vec::with_capacity(Self::DIM);
        for _ in 0..Self::DIM {
            state = state.wrapping_mul(0xbf58476d1ce4e5b9);
            state = state ^ (state >> 30);
            state = state.wrapping_mul(0x94d049bb133111eb);
            state = state ^ (state >> 27);
            state = state.wrapping_mul(0xbf58476d1ce4e5b9);
            let val = ((state & 0xFFFFFFFF) as f32 / 2147483648.0) - 1.0;
            values.push(val);
        }
        Self { values }
    }

    pub fn aggregate(vectors: &[Self]) -> Self {
        if vectors.is_empty() {
            return Self { values: vec![0.0; Self::DIM] };
        }
        let mut agg = vec![0.0; Self::DIM];
        for vec in vectors {
            for i in 0..Self::DIM {
                agg[i] += vec.values[i];
            }
        }
        let mut sum_sq = 0.0;
        for i in 0..Self::DIM {
            sum_sq += agg[i] * agg[i];
        }
        let norm = sum_sq.sqrt();
        if norm > 0.0 {
            for i in 0..Self::DIM {
                agg[i] /= norm;
            }
        }
        Self { values: agg }
    }

    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        let mut dot = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;
        for i in 0..Self::DIM {
            dot += self.values[i] * other.values[i];
            norm_a += self.values[i] * self.values[i];
            norm_b += other.values[i] * other.values[i];
        }
        let denom = (norm_a * norm_b).sqrt();
        if denom > 0.0 {
            dot / denom
        } else {
            0.0
        }
    }
}

pub fn jaccard_similarity(a: &[u64], b: &[u64]) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let set_a: BTreeSet<u64> = a.iter().copied().collect();
    let set_b: BTreeSet<u64> = b.iter().copied().collect();
    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();
    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

pub fn token_overlap_similarity(a: &[u64], b: &[u64]) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let set_a: BTreeSet<u64> = a.iter().copied().collect();
    let set_b: BTreeSet<u64> = b.iter().copied().collect();
    let overlap = set_a.intersection(&set_b).count();
    overlap as f32 / (a.len().min(b.len()) as f32)
}
