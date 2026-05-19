use crate::neuromodulator::Neuromodulator;
use crate::neuron::LifNeuron;
use crate::synapse::Synapse;
use std::error::Error;
use std::fmt;

/// Cortical Column Hierarchy & Predictive Coding Module
/// Mensimulasikan satu kolom korteks dengan dua lapisan:
/// - Lapisan L4 (Bottom-Up / Input Fitur Lokal)
/// - Lapisan L2/3 (Top-Down / Asosiasi Konteks Global)
///
/// Keduanya terhubung dua arah untuk melakukan Predictive Coding lokal.
pub struct CorticalColumn {
    pub num_l4: usize,
    pub num_l23: usize,
    pub l4_neurons: Vec<LifNeuron>,
    pub l23_neurons: Vec<LifNeuron>,
    /// Koneksi Feedforward (L4 -> L2/3)
    pub feedforward_synapses: Vec<Synapse>,
    /// Koneksi Feedback / Prediktif (L2/3 -> L4)
    pub feedback_synapses: Vec<Synapse>,
    /// Nilai akumulasi eror prediksi pada tick saat ini
    pub current_prediction_error: f32,
    /// Ambang batas arus prediksi untuk dianggap sebagai prediksi spike positif
    pub prediction_threshold: f32,
}

#[derive(Debug)]
pub enum ColumnStepError {
    InputDimensionMismatch { expected: usize, got: usize },
}

impl fmt::Display for ColumnStepError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputDimensionMismatch { expected, got } => {
                write!(
                    f,
                    "jumlah input spike tidak sesuai, expected {expected} namun dapat {got}"
                )
            }
        }
    }
}

impl Error for ColumnStepError {}

impl CorticalColumn {
    pub fn new(num_l4: usize, num_l23: usize) -> Self {
        // Lapisan L4 (Input): kebocoran cepat (0.7), threshold rendah (1.0), refraktori singkat (2 ticks)
        let l4_neurons = vec![LifNeuron::new(0.7, 1.0, 2); num_l4];

        // Lapisan L2/3 (Konteks): kebocoran lambat untuk memori temporal (0.9), threshold rendah untuk mudah terpicu (0.5), refraktori 4 ticks
        let l23_neurons = vec![LifNeuron::new(0.9, 0.5, 4); num_l23];

        // Inisialisasi koneksi feedforward acak (L4 -> L2/3) dengan bobot awal yang lebih besar
        let mut feedforward_synapses = Vec::new();
        for pre in 0..num_l4 {
            for post in 0..num_l23 {
                // Bobot awal yang lebih besar (0.4 - 0.8) agar mampu memicu L2/3
                let initial_w = 0.4 + ((pre + post) % 5) as f32 * 0.1;
                feedforward_synapses.push(Synapse::new(pre, post, initial_w));
            }
        }

        // Inisialisasi koneksi feedback acak (L2/3 -> L4)
        let mut feedback_synapses = Vec::new();
        for pre in 0..num_l23 {
            for post in 0..num_l4 {
                // Bobot awal acak kecil (0.2 - 0.5)
                let initial_w = 0.2 + ((pre + post) % 7) as f32 * 0.05;
                feedback_synapses.push(Synapse::new(pre, post, initial_w));
            }
        }

        Self {
            num_l4,
            num_l23,
            l4_neurons,
            l23_neurons,
            feedforward_synapses,
            feedback_synapses,
            current_prediction_error: 0.0,
            prediction_threshold: 0.5,
        }
    }

    /// Menjalankan simulasi satu langkah waktu (tick)
    /// `input_spikes` adalah array biner apakah neuron L4 menerima spike dari tokenizer
    pub fn step(
        &mut self,
        input_spikes: &[bool],
        t: usize,
        nm: &mut Neuromodulator,
        lr_ltp: f32,
        lr_ltd: f32,
    ) -> Result<(), ColumnStepError> {
        if input_spikes.len() != self.num_l4 {
            return Err(ColumnStepError::InputDimensionMismatch {
                expected: self.num_l4,
                got: input_spikes.len(),
            });
        }

        // 1. Tentukan status spike L4 berdasarkan input bottom-up.
        // Dalam biologis, L4 menerima input sensorik langsung.
        let mut l4_spiked = vec![false; self.num_l4];
        for i in 0..self.num_l4 {
            // Berikan arus besar (2.0) jika tokenizer mengirim spike pada neuron ini
            let current = if input_spikes[i] { 2.5 } else { 0.0 };
            // L4 neuron step (ACh memodulasi threshold L4 agar peka terhadap input baru)
            l4_spiked[i] = self.l4_neurons[i].step(current, t, nm.acetylcholine);
        }

        // 2. Hitung arus Feedforward dari L4 ke L2/3
        let mut l23_currents = vec![0.0f32; self.num_l23];
        for syn in &self.feedforward_synapses {
            if l4_spiked[syn.pre_idx] {
                // Tambahkan arus ke L2/3 proporsional terhadap bobot sinaptik
                l23_currents[syn.post_idx] += syn.weight;
            }
        }

        // 3. Jalankan satu langkah neuron L2/3 (Konteks)
        let mut l23_spiked = vec![false; self.num_l23];
        for j in 0..self.num_l23 {
            l23_spiked[j] = self.l23_neurons[j].step(l23_currents[j], t, nm.acetylcholine);
        }

        // 4. Hitung sinyal Prediksi Top-Down (Feedback) dari L2/3 ke L4
        // Lapisan L2/3 memproyeksikan kembali prediksi ke L4
        let mut l4_predictions = vec![0.0f32; self.num_l4];
        for syn in &self.feedback_synapses {
            if l23_spiked[syn.pre_idx] {
                l4_predictions[syn.post_idx] += syn.weight;
            }
        }

        // 5. Komparator: Hitung Eror Prediksi di Lapisan L4
        // Eror dihitung dari selisih antara input riil (spike) dan prediksi (arus feedback)
        let mut error_sum = 0.0;
        let mut correct_predictions = 0.0;

        for i in 0..self.num_l4 {
            let is_predicted = l4_predictions[i] >= self.prediction_threshold;
            let actual = l4_spiked[i];

            if actual && !is_predicted {
                // Surprise! Ada input tapi tidak terprediksi
                error_sum += 1.0;
            } else if !actual && is_predicted {
                // False alarm! Prediksi menyala tapi tidak ada input
                error_sum += 0.5;
            } else if actual && is_predicted {
                // Prediksi Tepat!
                correct_predictions += 1.0;
            }
        }

        self.current_prediction_error = error_sum;

        // 6. Neuromodulasi Feedback Loop
        if error_sum > 0.5 {
            // Jika eror tinggi, lepaskan Asetilkolin (ACh) untuk meningkatkan atensi/sensitivitas neuron
            nm.release_acetylcholine(0.2 * error_sum);
        } else if correct_predictions > 0.0 {
            // Jika prediksi tepat, lepaskan Dopamin sebagai bentuk "conformity reward"
            nm.release_dopamine(0.15 * correct_predictions);
        }

        // 7. Plastisitas Lokal (STDP): Perbarui bobot sinapsis Feedforward & Feedback
        // A. Feedforward Synapses (L4 -> L2/3)
        for syn in &mut self.feedforward_synapses {
            syn.decay_traces(0.92);
            let pre_s = l4_spiked[syn.pre_idx];
            let post_s = l23_spiked[syn.post_idx];
            syn.update_weight(pre_s, post_s, lr_ltp, lr_ltd, nm.dopamine);
        }

        // B. Feedback Synapses (L2/3 -> L4)
        for syn in &mut self.feedback_synapses {
            syn.decay_traces(0.92);
            let pre_s = l23_spiked[syn.pre_idx];
            let post_s = l4_spiked[syn.post_idx];
            syn.update_weight(pre_s, post_s, lr_ltp, lr_ltd, nm.dopamine);
        }

        Ok(())
    }

    /// Reset seluruh keadaan neuron (tapi pertahankan bobot sinapsis)
    pub fn reset_neurons(&mut self) {
        for neuron in &mut self.l4_neurons {
            neuron.reset_state();
        }
        for neuron in &mut self.l23_neurons {
            neuron.reset_state();
        }
        self.current_prediction_error = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::{ColumnStepError, CorticalColumn};
    use crate::neuromodulator::Neuromodulator;

    #[test]
    fn reject_input_with_invalid_dimension() {
        let mut column = CorticalColumn::new(3, 2);
        let mut neuromodulator = Neuromodulator::new(0.9);

        let error = column
            .step(&[true, false], 0, &mut neuromodulator, 0.05, 0.02)
            .expect_err("dimension mismatch must fail");

        assert!(matches!(
            error,
            ColumnStepError::InputDimensionMismatch {
                expected: 3,
                got: 2
            }
        ));
    }
}
