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
            // Berikan arus besar (2.5) jika tokenizer mengirim spike pada neuron ini
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

        // FITUR 1: GABAergic Lateral Inhibition (Inhibisi Lateral)
        // Meniru interkoneksi lokal sel GABAergik di neokorteks.
        // Neuron L2/3 yang menembak pada tick sebelumnya memicu pelepasan GABA,
        // yang melepaskan arus hambatan negatif pada neuron L2/3 lain di tick saat ini.
        let prev_l23_spikes = self.l23_neurons.iter().filter(|n| n.has_spiked).count();
        for j in 0..self.num_l23 {
            let other_spikes = if self.l23_neurons[j].has_spiked {
                prev_l23_spikes.saturating_sub(1)
            } else {
                prev_l23_spikes
            };
            let gaba_current = -0.3 * other_spikes as f32;
            l23_currents[j] += gaba_current;
        }

        // 3. Jalankan satu langkah neuron L2/3 (Konteks)
        let mut l23_spiked = vec![false; self.num_l23];
        for j in 0..self.num_l23 {
            l23_spiked[j] = self.l23_neurons[j].step(l23_currents[j], t, nm.acetylcholine);
        }

        // FITUR 3: Homeostatic Plasticity (Intrinsic Plasticity)
        // Penyesuaian lambat pada ambang batas neuron berdasarkan tingkat aktivitasnya.
        // Neuron yang baru saja menembak menaikkan threshold dasar agar tidak kelelahan/monopoli,
        // sedangkan neuron yang pasif perlahan menurunkan threshold agar lebih peka terhadap input berikutnya.
        for neuron in &mut self.l4_neurons {
            if neuron.has_spiked {
                neuron.v_threshold_base = (neuron.v_threshold_base + 0.02).min(2.5);
            } else {
                neuron.v_threshold_base = (neuron.v_threshold_base - 0.001).max(0.5);
            }
        }
        for neuron in &mut self.l23_neurons {
            if neuron.has_spiked {
                neuron.v_threshold_base = (neuron.v_threshold_base + 0.02).min(2.0);
            } else {
                neuron.v_threshold_base = (neuron.v_threshold_base - 0.001).max(0.2);
            }
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

        // FITUR 2: Structural Plasticity (Pruning & Sprouting Dinamis)
        // A. Pruning: Memotong sinapsis dengan bobot sangat kecil karena jarang diperkuat
        self.feedforward_synapses.retain(|syn| syn.weight > 0.05);
        self.feedback_synapses.retain(|syn| syn.weight > 0.05);

        // B. Sprouting: Menumbuhkan sinapsis baru dengan bobot minimal jika neuron pre dan post menembak bersamaan,
        // namun sebelumnya belum terhubung (atau telah terpotong)
        for pre in 0..self.num_l4 {
            if l4_spiked[pre] {
                for post in 0..self.num_l23 {
                    if l23_spiked[post] {
                        let exists = self.feedforward_synapses.iter().any(|syn| syn.pre_idx == pre && syn.post_idx == post);
                        if !exists {
                            self.feedforward_synapses.push(Synapse::new(pre, post, 0.1));
                        }
                    }
                }
            }
        }

        for pre in 0..self.num_l23 {
            if l23_spiked[pre] {
                for post in 0..self.num_l4 {
                    if l4_spiked[post] {
                        let exists = self.feedback_synapses.iter().any(|syn| syn.pre_idx == pre && syn.post_idx == post);
                        if !exists {
                            self.feedback_synapses.push(Synapse::new(pre, post, 0.1));
                        }
                    }
                }
            }
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

    #[test]
    fn test_neuro_mimetic_features() {
        let mut column = CorticalColumn::new(2, 2);
        let mut neuromodulator = Neuromodulator::new(0.9);

        // 1. Verifikasi GABAergic Lateral Inhibition (Inhibisi Lateral)
        // Paksa neuron L2/3 pertama menembak pada tick sebelumnya
        column.l23_neurons[0].has_spiked = true;
        // Step kolom: neuron L2/3 kedua harus terhambat oleh GABA negatif
        column.step(&[true, false], 1, &mut neuromodulator, 0.05, 0.02).unwrap();

        // 2. Verifikasi Homeostatic Plasticity (Intrinsic Plasticity)
        // Catat threshold dasar sebelum langkah berikutnya
        let l4_0_base_init = column.l4_neurons[0].v_threshold_base;
        let l4_1_base_init = column.l4_neurons[1].v_threshold_base;

        // Step kolom lagi dengan penembakan di L4[0]
        column.step(&[true, false], 2, &mut neuromodulator, 0.05, 0.02).unwrap();

        // Neuron L4[0] (aktif) threshold dasar naik/turun sesuai status tembakan
        if column.l4_neurons[0].has_spiked {
            assert!(column.l4_neurons[0].v_threshold_base > l4_0_base_init);
        } else {
            assert!(column.l4_neurons[0].v_threshold_base < l4_0_base_init);
        }
        // Neuron L4[1] (pasif) threshold dasar harus turun agar lebih peka
        assert!(column.l4_neurons[1].v_threshold_base < l4_1_base_init);

        // 3. Verifikasi Structural Plasticity (Pruning)
        // Set bobot salah satu sinapsis feedforward ke nilai sangat rendah
        column.feedforward_synapses[0].weight = 0.01;
        let count_before = column.feedforward_synapses.len();
        // Lakukan pemrosesan step yang memicu pruning
        column.step(&[false, false], 3, &mut neuromodulator, 0.05, 0.02).unwrap();
        let count_after = column.feedforward_synapses.len();
        // Jumlah sinapsis harus berkurang karena sinapsis berbobot 0.01 telah dipangkas (pruned)
        assert!(count_after < count_before, "Sinapsis lemah (bobot <= 0.05) harus dipangkas");
    }
}
