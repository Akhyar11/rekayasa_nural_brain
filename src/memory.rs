use crate::cortical::CorticalColumn;
use crate::neuromodulator::Neuromodulator;
use crate::synapse::Synapse;
use std::collections::VecDeque;

/// Modul Hipokampus (Memori Jangka Pendek & Fast-Binding)
/// Hipokampus merekam representasi temporal dinamis dari Lapisan L2/3 Neokorteks
/// secara instan selama fase aktif (bangun).
pub struct Hippocampus {
    /// Buffer temporal yang merekam deretan spike L2/3 secara runtun waktu
    pub episodic_buffer: VecDeque<Vec<bool>>,
    /// Kapasitas maksimum buffer memori jangka pendek
    pub capacity: usize,
}

impl Hippocampus {
    pub fn new(capacity: usize) -> Self {
        Self {
            episodic_buffer: VecDeque::new(),
            capacity,
        }
    }

    /// Merekam pola penembakan neuron (spike) dari L2/3
    pub fn record(&mut self, l23_spikes: Vec<bool>) {
        if self.episodic_buffer.len() >= self.capacity {
            self.episodic_buffer.pop_front();
        }
        self.episodic_buffer.push_back(l23_spikes);
    }

    /// Membersihkan memori jangka pendek (setelah konsolidasi sukses)
    pub fn clear(&mut self) {
        self.episodic_buffer.clear();
    }
}

/// Dual-Memory System Manager
/// Mengelola proses konsolidasi memori jangka pendek dari Hipokampus
/// ke dalam sinapsis jangka panjang Neokorteks melalui simulasi tidur (Sleep Replay).
pub struct MemoryConsolidator;

impl Default for MemoryConsolidator {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryConsolidator {
    pub fn new() -> Self {
        Self
    }

    /// Fase Replay / Tidur (Sleep-Consolidation Phase)
    /// Memutar ulang memori jangka pendek yang terekam di Hipokampus berulang kali dengan kecepatan tinggi.
    /// Pola ini memicu plastisitas lambat di Neokorteks (Cortical Column) tanpa merusak pengetahuan lama.
    pub fn consolidate(
        &self,
        column: &mut CorticalColumn,
        nm: &mut Neuromodulator,
        hippocampus: &mut Hippocampus,
        epochs: usize,
    ) -> usize {
        if hippocampus.episodic_buffer.is_empty() {
            return 0; // Tidak ada memori untuk dikonsolidasi
        }

        let mut replay_ticks = 0;

        // Neuromodulasi dalam keadaan tidur: Asetilkolin (ACh) rendah, Dopamin (DA) sedikit aktif
        // Ini merepresentasikan "Non-REM / REM Sleep" di mana plastisitas tetap berjalan tapi atensi sensorik mati
        nm.acetylcholine = 0.05;
        nm.dopamine = 0.3;
        column.reset_neurons();

        for _epoch in 0..epochs {
            for pattern in &hippocampus.episodic_buffer {
                // Di fase tidur, tidak ada input dari luar (Tokenizer nonaktif).
                // Kita merekonstruksi keadaan korteks dengan langsung menembakkan arus ke L2/3
                // berdasarkan pola hipokampus, lalu membiarkan STDP menyelaraskan sinapsis.
                let mut l23_spikes = vec![false; column.num_l23];
                for (j, l23_spike) in l23_spikes.iter_mut().enumerate().take(column.num_l23) {
                    // Gunakan threshold biologis normal (tanpa penurunan ACh)
                    let current = if pattern.get(j).copied().unwrap_or(false) {
                        3.0
                    } else {
                        0.0
                    };
                    *l23_spike =
                        column.l23_neurons[j].step(current, replay_ticks, nm.acetylcholine);
                }

                // Hitung feedback prediksi L2/3 ke L4
                let mut l4_predictions = vec![0.0f32; column.num_l4];
                for syn in &column.feedback_synapses {
                    if l23_spikes[syn.pre_idx] {
                        l4_predictions[syn.post_idx] += syn.weight;
                    }
                }

                // Di fase tidur, L4 menyala murni karena stimulasi feedback top-down (mimpi/replay)
                let mut l4_spikes = vec![false; column.num_l4];
                for i in 0..column.num_l4 {
                    l4_spikes[i] = column.l4_neurons[i].step(
                        l4_predictions[i],
                        replay_ticks,
                        nm.acetylcholine,
                    );
                }

                // Jalankan pembelajaran STDP lambat pada sinapsis feedforward & feedback
                // Kita gunakan learning rate yang lebih kecil saat tidur agar perubahan bersifat inkremental/stabil
                let sleep_lr_ltp = 0.01;
                let sleep_lr_ltd = 0.005;

                for syn in &mut column.feedforward_synapses {
                    syn.decay_traces(0.95);
                    let pre_s = l4_spikes[syn.pre_idx];
                    let post_s = l23_spikes[syn.post_idx];
                    syn.update_weight(pre_s, post_s, sleep_lr_ltp, sleep_lr_ltd, nm.dopamine);
                }

                for syn in &mut column.feedback_synapses {
                    syn.decay_traces(0.95);
                    let pre_s = l23_spikes[syn.pre_idx];
                    let post_s = l4_spikes[syn.post_idx];
                    syn.update_weight(pre_s, post_s, sleep_lr_ltp, sleep_lr_ltd, nm.dopamine);
                }

                // FITUR 2: Structural Plasticity (Pruning & Sprouting Dinamis) saat Tidur
                column.feedforward_synapses.retain(|syn| syn.weight > 0.05);
                column.feedback_synapses.retain(|syn| syn.weight > 0.05);

                for pre in 0..column.num_l4 {
                    if l4_spikes[pre] {
                        for post in 0..column.num_l23 {
                            if l23_spikes[post] {
                                let exists = column.feedforward_synapses.iter().any(|syn| syn.pre_idx == pre && syn.post_idx == post);
                                if !exists {
                                    column.feedforward_synapses.push(Synapse::new(pre, post, 0.1));
                                }
                            }
                        }
                    }
                }

                for pre in 0..column.num_l23 {
                    if l23_spikes[pre] {
                        for post in 0..column.num_l4 {
                            if l4_spikes[post] {
                                let exists = column.feedback_synapses.iter().any(|syn| syn.pre_idx == pre && syn.post_idx == post);
                                if !exists {
                                    column.feedback_synapses.push(Synapse::new(pre, post, 0.1));
                                }
                            }
                        }
                    }
                }

                // FITUR 3: Homeostatic Plasticity saat Tidur
                for neuron in &mut column.l4_neurons {
                    if neuron.has_spiked {
                        neuron.v_threshold_base = (neuron.v_threshold_base + 0.02).min(2.5);
                    } else {
                        neuron.v_threshold_base = (neuron.v_threshold_base - 0.001).max(0.5);
                    }
                }
                for neuron in &mut column.l23_neurons {
                    if neuron.has_spiked {
                        neuron.v_threshold_base = (neuron.v_threshold_base + 0.02).min(2.0);
                    } else {
                        neuron.v_threshold_base = (neuron.v_threshold_base - 0.001).max(0.2);
                    }
                }

                replay_ticks += 1;
            }
        }

        // Bersihkan Hipokampus setelah berhasil dikonsolidasikan ke Neokorteks
        hippocampus.clear();
        column.reset_neurons();
        nm.acetylcholine = 0.0;
        nm.dopamine = 0.0;

        replay_ticks
    }
}

#[cfg(test)]
mod tests {
    use super::Hippocampus;

    #[test]
    fn record_keeps_fifo_capacity() {
        let mut hippocampus = Hippocampus::new(2);

        hippocampus.record(vec![true, false]);
        hippocampus.record(vec![false, true]);
        hippocampus.record(vec![true, true]);

        assert_eq!(hippocampus.episodic_buffer.len(), 2);
        assert_eq!(
            hippocampus.episodic_buffer.front(),
            Some(&vec![false, true])
        );
        assert_eq!(hippocampus.episodic_buffer.back(), Some(&vec![true, true]));
    }
}
