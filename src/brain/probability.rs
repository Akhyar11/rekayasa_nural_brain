pub fn calculate_smoothed_probability(
    count: u64,
    total_count: u64,
    num_categories: usize,
    alpha: f32,
) -> f32 {
    let smoothed_numerator = count as f32 + alpha;
    let smoothed_denominator = total_count as f32 + (num_categories as f32 * alpha);
    smoothed_numerator / smoothed_denominator
}

pub fn entropy(probabilities: &[f32]) -> f32 {
    let mut h = 0.0;
    for &p in probabilities {
        if p > 0.0 {
            h -= p * p.ln();
        }
    }
    h
}
