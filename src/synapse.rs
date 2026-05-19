/// Spike-Timing-Dependent Plasticity (STDP) Synapse
/// Menghubungkan neuron pre-sinaptik ke neuron post-sinaptik dengan
/// pembaruan bobot secara asinkron berdasarkan selisih waktu lonjakan sinyal (spikes).
#[derive(Clone, Debug)]
pub struct Synapse {
    /// Indeks neuron asal (pre-synaptic)
    pub pre_idx: usize,
    /// Indeks neuron tujuan (post-synaptic)
    pub post_idx: usize,
    /// Kekuatan koneksi/bobot sinaptik (W)
    pub weight: f32,
    /// Nilai jejak pre-sinaptik (x_trace) yang meluruh seiring waktu
    pub pre_trace: f32,
    /// Nilai jejak post-sinaptik (y_trace) yang meluruh seiring waktu
    pub post_trace: f32,
}

impl Synapse {
    pub fn new(pre_idx: usize, post_idx: usize, initial_weight: f32) -> Self {
        Self {
            pre_idx,
            post_idx,
            weight: initial_weight.clamp(0.01, 5.0),
            pre_trace: 0.0,
            post_trace: 0.0,
        }
    }

    /// Melakukan peluruhan jejak (trace decay) pada setiap langkah waktu diskret
    /// `decay_factor` biasanya berkisar antara 0.9 hingga 0.95
    pub fn decay_traces(&mut self, decay_factor: f32) {
        let decay_factor = decay_factor.clamp(0.0, 1.0);
        self.pre_trace *= decay_factor;
        self.post_trace *= decay_factor;
    }

    /// Memperbarui bobot sinapsis menggunakan aturan STDP Hebbian lokal.
    /// - `pre_spiked`: Apakah neuron asal (pre) menembak pada langkah waktu ini.
    /// - `post_spiked`: Apakah neuron tujuan (post) menembak pada langkah waktu ini.
    /// - `lr_ltp`: Learning rate untuk Long-Term Potentiation (penguatan).
    /// - `lr_ltd`: Learning rate untuk Long-Term Depression (pelemahan).
    /// - `dopamine`: Konsentrasi neuromodulator Dopamin (0.0 - 1.0) untuk memperkuat/memodulasi plastisitas.
    pub fn update_weight(
        &mut self,
        pre_spiked: bool,
        post_spiked: bool,
        lr_ltp: f32,
        lr_ltd: f32,
        dopamine: f32,
    ) -> f32 {
        let mut delta_w = 0.0;

        // 1. Jika neuron pre-sinaptik menembak (Pre-Spike)
        if pre_spiked {
            // Naikkan trace pre-sinaptik
            self.pre_trace += 1.0;
            if self.pre_trace > 2.0 {
                self.pre_trace = 2.0; // Batasi saturasi trace
            }

            // Jika pre-spike terjadi SETELAH post-spike (post_trace > 0),
            // maka terjadi pelemahan bobot (LTD)
            // Dopamin bertindak sebagai sinyal kepuasan: jika dopamin tinggi, LTD sedikit dikurangi (mempertahankan koneksi positif)
            let ltd_modulation = 1.0 - 0.5 * dopamine;
            delta_w -= lr_ltd * self.post_trace * ltd_modulation;
        }

        // 2. Jika neuron post-sinaptik menembak (Post-Spike)
        if post_spiked {
            // Naikkan trace post-sinaptik
            self.post_trace += 1.0;
            if self.post_trace > 2.0 {
                self.post_trace = 2.0;
            }

            // Jika pre-spike terjadi SEBELUM post-spike (pre_trace > 0),
            // maka terjadi penguatan bobot (LTP)
            // Neuromodulasi: Dopamin memperkuat LTP (High reward = super strong memory binding)
            let ltp_modulation = 1.0 + 2.0 * dopamine;
            delta_w += lr_ltp * self.pre_trace * ltp_modulation;
        }

        // Terapkan perubahan bobot dan batasi (clipping) agar tetap stabil (biologically bounded)
        // Kita batasi bobot antara 0.01 (koneksi minimal) dan 5.0 (koneksi super kuat)
        self.weight = (self.weight + delta_w).clamp(0.01, 5.0);

        delta_w
    }
}

#[cfg(test)]
mod tests {
    use super::Synapse;

    #[test]
    fn weight_is_clamped_during_updates() {
        let mut synapse = Synapse::new(0, 1, 10.0);
        synapse.pre_trace = 2.0;
        synapse.post_trace = 2.0;

        synapse.update_weight(true, true, 1.0, 1.0, 1.0);

        assert!((0.01..=5.0).contains(&synapse.weight));
    }
}
