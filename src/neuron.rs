/// Leaky Integrate-and-Fire (LIF) Neuron Model
/// Neuron biologis dasar yang memiliki kebocoran potensial membran,
/// batas ambang tembakan (threshold) dinamis, dan masa refraktori.
#[derive(Clone, Debug)]
pub struct LifNeuron {
    /// Potensial membran saat ini (V)
    pub v: f32,
    /// Potensial istirahat (Resting Potential)
    pub v_rest: f32,
    /// Potensial setelah menembak (Reset Potential)
    pub v_reset: f32,
    /// Ambang batas tembakan dasar (Base Firing Threshold)
    pub v_threshold_base: f32,
    /// Ambang batas tembakan saat ini (bisa diturunkan oleh Asetilkolin)
    pub v_threshold: f32,
    /// Konstanta peluruhan membran (Leak Rate, antara 0.0 dan 1.0)
    pub leak_factor: f32,
    /// Durasi masa refraktori (dalam ticks/langkah waktu) setelah menembak
    pub refractory_period: usize,
    /// Sisa langkah waktu dalam masa refraktori
    pub refractory_steps_left: usize,
    /// Apakah neuron menembak pada langkah waktu ini
    pub has_spiked: bool,
    /// Waktu langkah terakhir neuron menembak (untuk kalkulasi STDP)
    pub last_spike_time: Option<usize>,
}

impl LifNeuron {
    pub fn new(leak_factor: f32, threshold: f32, refractory_period: usize) -> Self {
        Self {
            v: 0.0,
            v_rest: 0.0,
            v_reset: 0.0,
            v_threshold_base: threshold,
            v_threshold: threshold,
            leak_factor: leak_factor.clamp(0.0, 1.0),
            refractory_period,
            refractory_steps_left: 0,
            has_spiked: false,
            last_spike_time: None,
        }
    }

    /// Memperbarui keadaan neuron berdasarkan arus input dinamis pada langkah waktu `t`
    /// `acetylcholine` adalah konsentrasi neuromodulator ACh (0.0 - 1.0) yang menurunkan ambang batas
    pub fn step(&mut self, input_current: f32, t: usize, acetylcholine: f32) -> bool {
        self.has_spiked = false;

        // Neuromodulasi: ACh menurunkan ambang batas tembakan neuron agar lebih sensitif terhadap input
        // Maksimal menurunkan threshold hingga 40% dari batas dasar
        let acetylcholine = acetylcholine.clamp(0.0, 1.0);
        self.v_threshold = self.v_threshold_base * (1.0 - 0.4 * acetylcholine);

        // Jika berada dalam masa refraktori, neuron tetap di potensial reset dan tidak bisa menembak
        if self.refractory_steps_left > 0 {
            self.refractory_steps_left -= 1;
            self.v = self.v_reset;
            return false;
        }

        // Terapkan kebocoran membran potensial dan tambahkan arus input
        self.v = (self.v - self.v_rest) * self.leak_factor + self.v_rest + input_current;

        // Cek apakah potensial membran melebihi ambang batas
        if self.v >= self.v_threshold {
            self.has_spiked = true;
            self.v = self.v_reset; // Reset potensial membran
            self.refractory_steps_left = self.refractory_period; // Masuk masa refraktori
            self.last_spike_time = Some(t);
        }

        self.has_spiked
    }

    /// Reset seluruh keadaan neuron (untuk simulasi baru atau fase replay)
    pub fn reset_state(&mut self) {
        self.v = self.v_rest;
        self.refractory_steps_left = 0;
        self.has_spiked = false;
        self.last_spike_time = None;
    }
}

#[cfg(test)]
mod tests {
    use super::LifNeuron;

    #[test]
    fn neuron_spikes_and_honors_refractory_period() {
        let mut neuron = LifNeuron::new(0.9, 1.0, 2);

        assert!(neuron.step(1.2, 0, 0.0));
        assert!(!neuron.step(1.2, 1, 0.0));
        assert!(!neuron.step(1.2, 2, 0.0));
        assert!(neuron.step(1.2, 3, 0.0));
    }
}
