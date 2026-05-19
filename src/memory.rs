use crate::cortical::CorticalColumn;
use crate::neuromodulator::Neuromodulator;

/// Modul Hipokampus (Memori Jangka Pendek & Fast-Binding)
/// Hipokampus merekam representasi temporal dinamis dari Lapisan L2/3 Neokorteks
/// secara instan selama fase aktif (bangun).
pub struct Hippocampus {
    /// Buffer temporal yang merekam deretan spike L2/3 secara runtun waktu
    pub episodic_buffer: Vec<Vec<bool>>,
    /// Kapasitas maksimum buffer memori jangka pendek
    pub capacity: usize,
}

impl Hippocampus {
    pub fn new(capacity: usize) -> Self {
        Self {
            episodic_buffer: Vec::new(),
            capacity,
        }
    }

    /// Merekam pola penembakan neuron (spike) dari L2/3
    pub fn record(&mut self, l23_spikes: Vec<bool>) {
        if self.episodic_buffer.len() >= self.capacity {
            // Jika melebihi kapasitas, buang memori paling usang (FIFO)
            self.episodic_buffer.remove(0);
        }
        self.episodic_buffer.push(l23_spikes);
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

        for _epoch in 0..epochs {
            for pattern in &hippocampus.episodic_buffer {
                // Di fase tidur, tidak ada input dari luar (Tokenizer nonaktif).
                // Kita merekonstruksi keadaan korteks dengan langsung menembakkan arus ke L2/3
                // berdasarkan pola hipokampus, lalu membiarkan STDP menyelaraskan sinapsis.
                let mut l23_spikes = vec![false; column.num_l23];
                for j in 0..column.num_l23 {
                    // Gunakan threshold biologis normal (tanpa penurunan ACh)
                    let current = if pattern[j] { 3.0 } else { 0.0 };
                    l23_spikes[j] = column.l23_neurons[j].step(current, replay_ticks, nm.acetylcholine);
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
                    l4_spikes[i] = column.l4_neurons[i].step(l4_predictions[i], replay_ticks, nm.acetylcholine);
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

                replay_ticks += 1;
            }
        }

        // Bersihkan Hipokampus setelah berhasil dikonsolidasikan ke Neokorteks
        hippocampus.clear();

        replay_ticks
    }
}
