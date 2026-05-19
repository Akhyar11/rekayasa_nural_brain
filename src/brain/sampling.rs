use super::generation::TokenCandidate;
use super::config::GenerationConfig;

pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn new(seed: Option<u64>) -> Self {
        let actual_seed = seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64
        });
        Self {
            state: actual_seed ^ 0x5555555555555555,
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() & 0xFFFFFFFF) as f32 / 4294967296.0
    }
}

pub fn sample_next_token(
    candidates: &[TokenCandidate],
    config: &GenerationConfig,
) -> Option<u64> {
    if candidates.is_empty() {
        return None;
    }

    let k = config.top_k.max(1);
    let mut top_k_candidates = candidates.to_vec();
    if top_k_candidates.len() > k {
        top_k_candidates.truncate(k);
    }

    let total_p: f32 = top_k_candidates.iter().map(|c| c.probability).sum();
    if total_p <= 0.0 {
        return Some(top_k_candidates[0].token_id);
    }
    for c in &mut top_k_candidates {
        c.probability /= total_p;
    }

    let mut cumulative_p = 0.0;
    let mut top_p_candidates = Vec::new();
    for c in top_k_candidates {
        cumulative_p += c.probability;
        top_p_candidates.push(c);
        if cumulative_p >= config.top_p {
            break;
        }
    }

    let total_p: f32 = top_p_candidates.iter().map(|c| c.probability).sum();
    if total_p <= 0.0 {
        return Some(top_p_candidates[0].token_id);
    }
    for c in &mut top_p_candidates {
        c.probability /= total_p;
    }

    let mut rng = SimpleRng::new(config.randomness_seed);
    let mut r = rng.next_f32() * total_p;
    for c in &top_p_candidates {
        if r < c.probability {
            return Some(c.token_id);
        }
        r -= c.probability;
    }

    Some(top_p_candidates[0].token_id)
}
