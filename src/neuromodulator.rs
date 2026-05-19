/// Neuromodulator Dynamics Module
/// Mensimulasikan pelepasan senyawa neuromodulator (Asetilkolin dan Dopamin)
/// untuk mengatur perilaku plastisitas dan ambang batas penembakan secara real-time.
#[derive(Clone, Debug)]
pub struct Neuromodulator {
    /// Asetilkolin (ACh): Mengatur tingkat fokus / atensi.
    /// ACh yang tinggi menurunkan ambang batas neuron agar lebih peka terhadap informasi baru (kejutan).
    pub acetylcholine: f32,
    /// Dopamin (DA): Mengatur sistem reward / penguatan memori.
    /// Dopamin yang tinggi memperkuat LTP pada aturan STDP.
    pub dopamine: f32,
    /// Laju peluruhan neuromodulator per langkah waktu (misal 0.90)
    pub decay_rate: f32,
}

impl Neuromodulator {
    pub fn new(decay_rate: f32) -> Self {
        Self {
            acetylcholine: 0.0,
            dopamine: 0.0,
            decay_rate,
        }
    }

    /// Melakukan pembaruan langkah waktu diskret (peluruhan senyawa kimiawi)
    pub fn step(&mut self) {
        self.acetylcholine *= self.decay_rate;
        self.dopamine *= self.decay_rate;

        // Jaga agar nilai tidak terlalu mendekati nol mutlak (menghindari underflow)
        if self.acetylcholine < 0.001 { self.acetylcholine = 0.0; }
        if self.dopamine < 0.001 { self.dopamine = 0.0; }
    }

    /// Memicu pelepasan Asetilkolin ketika terdeteksi "kejutan informasi" atau Prediction Error yang tinggi
    pub fn release_acetylcholine(&mut self, amount: f32) {
        self.acetylcholine = (self.acetylcholine + amount).clamp(0.0, 1.0);
    }

    /// Memicu pelepasan Dopamin ketika model berhasil memprediksi dengan benar atau menerima konfirmasi eksternal
    pub fn release_dopamine(&mut self, amount: f32) {
        self.dopamine = (self.dopamine + amount).clamp(0.0, 1.0);
    }

    /// Mengembalikan status visual untuk dashboard
    pub fn status_string(&self) -> String {
        format!(
            "ACh (Atensi): {:.2} | DA (Reward): {:.2}",
            self.acetylcholine, self.dopamine
        )
    }
}
