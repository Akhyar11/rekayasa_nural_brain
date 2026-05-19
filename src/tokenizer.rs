/// Temporal Spiking Tokenizer
/// Mengubah teks input menjadi rangkaian sinyal spike (pulsa biner)
/// dalam domain waktu untuk disimulasikan oleh jaringan saraf biologis.
pub struct SpikingTokenizer {
    pub vocab_size: usize,
}

impl SpikingTokenizer {
    pub fn new() -> Self {
        // Kita menggunakan karakter a-z (26) + spasi (1) = 27 input neuron
        Self { vocab_size: 27 }
    }

    /// Mengubah karakter menjadi index neuron (0-26)
    fn char_to_index(&self, c: char) -> Option<usize> {
        let c_lower = c.to_ascii_lowercase();
        if c_lower.is_ascii_alphabetic() {
            Some((c_lower as u8 - b'a') as usize)
        } else if c_lower == ' ' {
            Some(26) // Spasi dipetakan ke neuron index 26
        } else {
            None // Abaikan karakter lain
        }
    }

    /// Mengubah string input menjadi matriks Spikes [TimeSteps][VocabSize]
    /// Setiap karakter dipicu pada jendela waktu tertentu dengan latensi.
    pub fn encode(&self, text: &str, steps_per_char: usize) -> Vec<Vec<bool>> {
        let chars: Vec<char> = text.chars().collect();
        let total_steps = chars.len() * steps_per_char;
        let mut spike_matrix = vec![vec![false; self.vocab_size]; total_steps];

        for (i, &c) in chars.iter().enumerate() {
            if let Some(neuron_idx) = self.char_to_index(c) {
                // Berikan spike temporal: neuron akan menyala pada awal interval karakter
                // ditambahkan latensi tertentu berdasarkan karakter itu sendiri (biologically plausible latency encoding)
                let start_step = i * steps_per_char;
                // Latensi: karakter lebih awal dalam alfabet menyala sedikit lebih awal
                let latency = (neuron_idx % 3) * (steps_per_char / 4);
                let spike_step = start_step + latency;
                if spike_step < total_steps {
                    spike_matrix[spike_step][neuron_idx] = true;
                }
            }
        }

        spike_matrix
    }

    /// Mengubah kembali index neuron menjadi karakter
    pub fn decode_char(&self, idx: usize) -> char {
        if idx < 26 {
            (b'a' + idx as u8) as char
        } else {
            ' '
        }
    }
}
