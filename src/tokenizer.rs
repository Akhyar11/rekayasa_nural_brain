use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

/// Temporal Spiking Tokenizer
/// Mengubah teks input menjadi rangkaian sinyal spike (pulsa biner)
/// dalam domain waktu untuk disimulasikan oleh jaringan saraf biologis.
#[derive(Clone, Debug)]
pub struct SpikingTokenizer {
    pub vocab_size: usize,
}

impl Default for SpikingTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TokenizerError {
    EmptyInput,
    ZeroStepsPerChar,
    UnsupportedCharacters(Vec<char>),
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "teks input tidak boleh kosong"),
            Self::ZeroStepsPerChar => write!(f, "steps-per-char harus lebih besar dari 0"),
            Self::UnsupportedCharacters(chars) => {
                write!(f, "karakter tidak didukung: {:?}", chars)
            }
        }
    }
}

impl Error for TokenizerError {}

impl SpikingTokenizer {
    pub fn new() -> Self {
        // Kita menggunakan karakter a-z (26) + spasi (1) = 27 input neuron
        Self { vocab_size: 27 }
    }

    /// Mengubah karakter menjadi index neuron (0-26)
    pub fn char_to_index(&self, c: char) -> Option<usize> {
        let c_lower = c.to_ascii_lowercase();
        if c_lower.is_ascii_alphabetic() {
            Some((c_lower as u8 - b'a') as usize)
        } else if c_lower == ' ' {
            Some(26) // Spasi dipetakan ke neuron index 26
        } else {
            None
        }
    }

    pub fn validate_text(&self, text: &str) -> Result<(), TokenizerError> {
        if text.trim().is_empty() {
            return Err(TokenizerError::EmptyInput);
        }

        let unsupported: BTreeSet<char> = text
            .chars()
            .filter(|character| self.char_to_index(*character).is_none())
            .collect();

        if unsupported.is_empty() {
            Ok(())
        } else {
            Err(TokenizerError::UnsupportedCharacters(
                unsupported.into_iter().collect(),
            ))
        }
    }

    /// Mengubah string input menjadi matriks Spikes [TimeSteps][VocabSize]
    /// Setiap karakter dipicu pada jendela waktu tertentu dengan latensi.
    pub fn encode(
        &self,
        text: &str,
        steps_per_char: usize,
    ) -> Result<Vec<Vec<bool>>, TokenizerError> {
        if steps_per_char == 0 {
            return Err(TokenizerError::ZeroStepsPerChar);
        }

        self.validate_text(text)?;

        let chars: Vec<char> = text.chars().collect();
        let total_steps = chars.len() * steps_per_char;
        let mut spike_matrix = vec![vec![false; self.vocab_size]; total_steps];

        for (index, character) in chars.iter().enumerate() {
            let neuron_idx = self
                .char_to_index(*character)
                .expect("validated input should only contain supported characters");
            let start_step = index * steps_per_char;
            let latency = (neuron_idx % 3) * (steps_per_char / 4);
            let spike_step = start_step + latency;
            if spike_step < total_steps {
                spike_matrix[spike_step][neuron_idx] = true;
            }
        }

        Ok(spike_matrix)
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

#[cfg(test)]
mod tests {
    use super::{SpikingTokenizer, TokenizerError};

    #[test]
    fn encode_supported_text() {
        let tokenizer = SpikingTokenizer::new();
        let spikes = tokenizer
            .encode("ab c", 4)
            .expect("text should be supported");

        assert_eq!(spikes.len(), 16);
        assert!(spikes[0][0]);
    }

    #[test]
    fn reject_unsupported_characters() {
        let tokenizer = SpikingTokenizer::new();
        let error = tokenizer
            .encode("brain!", 4)
            .expect_err("unsupported character must fail");

        assert_eq!(error, TokenizerError::UnsupportedCharacters(vec!['!']));
    }

    #[test]
    fn reject_zero_steps() {
        let tokenizer = SpikingTokenizer::new();
        let error = tokenizer
            .encode("brain", 0)
            .expect_err("zero steps must fail");

        assert_eq!(error, TokenizerError::ZeroStepsPerChar);
    }
}
