use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use super::error::BrainError;

// BPE Special tokens matching Oxide-JS
pub const PAD_TOKEN: &str = "<PAD>";
pub const UNK_TOKEN: &str = "<UNK>";
pub const BOS_TOKEN: &str = "<BOS>";
pub const EOS_TOKEN: &str = "<EOS>";
pub const WORD_BOUNDARY: &str = "▁";
pub const PAIR_SEPARATOR: char = '\0';

// Default bootstrap corpus for auto-training BPE tokenizer on startup
pub const DEFAULT_BOOTSTRAP_CORPUS: &[&str] = &[
    "saya suka kopi hitam panas",
    "teh manis dingin segar sekali",
    "siapa kamu saya adalah brain kognitif",
    "bagaimana cara kamu belajar dari interaksi",
    "hari ini cuaca sangat cerah dan bagus",
    "apa yang bisa saya bantu untuk anda",
    "terima kasih banyak atas bantuan dan informasinya",
    "sama sama senang bisa membantu anda",
    "selamat pagi siang sore malam",
    "apakah kamu mengerti bahasa indonesia dengan baik",
    "ya saya belajar memahami konteks dan relasi semantik",
    "kopi dan teh adalah jenis minuman populer",
    "saya tidak akan berbohong jika tidak tahu jawabannya",
];

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum TokenLevel {
    Sensor,
    Word,
    Phrase,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TokenEntry {
    pub node_id: u64,
    pub text: String,
    pub level: TokenLevel,
    pub occurrence_count: u64,
    pub created_at: u64,
    pub last_used_at: u64,
}

#[derive(Clone, Debug, Default)]
pub struct TrieNode {
    pub children: BTreeMap<char, TrieNode>,
    pub token_id: Option<u64>,
    pub level: Option<TokenLevel>,
}

#[derive(Clone, Debug, Default)]
pub struct TokenTrie {
    pub root: TrieNode,
}

impl TokenTrie {
    pub fn insert(&mut self, text: &str, token_id: u64, level: TokenLevel) {
        let mut current = &mut self.root;
        for c in text.chars() {
            current = current.children.entry(c).or_default();
        }
        current.token_id = Some(token_id);
        current.level = Some(level);
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptiveTokenizer {
    pub lookup: BTreeMap<String, u64>,
    pub entries: BTreeMap<u64, TokenEntry>,
    pub surface_counts: BTreeMap<String, u64>,
    pub merges: Vec<(String, String)>,
    pub vocab_size: usize,
    pub min_frequency: usize,

    #[serde(skip, default)]
    pub trie: TokenTrie,
}

impl Default for AdaptiveTokenizer {
    fn default() -> Self {
        let mut tok = Self {
            lookup: BTreeMap::new(),
            entries: BTreeMap::new(),
            surface_counts: BTreeMap::new(),
            merges: Vec::new(),
            vocab_size: 1000,
            min_frequency: 1,
            trie: TokenTrie::default(),
        };
        // Auto-train BPE on default Indonesian bootstrap corpus
        let corpus: Vec<String> = DEFAULT_BOOTSTRAP_CORPUS.iter().map(|s| s.to_string()).collect();
        tok.train(&corpus);
        tok
    }
}

struct WordSymbols {
    symbols: Vec<String>,
    freq: usize,
}

fn pre_tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for c in text.chars() {
        if c.is_whitespace() {
            if !current.is_empty() {
                tokens.push(current.clone());
                current.clear();
            }
        } else {
            current.push(c);
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

fn create_initial_symbols(token: &str) -> Vec<String> {
    token.chars().map(|c| c.to_string()).collect()
}

fn is_mergeable_token(token: &str, special_tokens: &[String]) -> bool {
    if token.is_empty() {
        return false;
    }
    if special_tokens.iter().any(|t| t == token) || token.starts_with("<UNUSED_") || token.starts_with("<RESERVED_") {
        return true;
    }
    let without_boundary = token.replace(WORD_BOUNDARY, "");
    if without_boundary.is_empty() {
        return true;
    }
    for c in without_boundary.chars() {
        if c.is_ascii_punctuation() || c.is_whitespace() || is_math_or_emoji(c) {
            return false;
        }
    }
    true
}

fn is_math_or_emoji(c: char) -> bool {
    let cp = c as u32;
    (cp >= 0x1f000 && cp <= 0x1faff) || (cp >= 0x2600 && cp <= 0x27bf) || (cp >= 0x2070 && cp <= 0x209f) || (cp >= 0x2190 && cp <= 0x21ff) || (cp >= 0x2200 && cp <= 0x22ff)
}

fn apply_merge_in_place(symbols: &mut Vec<String>, left: &str, right: &str, merged: &str) -> bool {
    let mut read_idx = 0;
    let mut write_idx = 0;
    let mut changed = false;
    if symbols.is_empty() {
        return false;
    }
    let len = symbols.len();
    while read_idx < len {
        if read_idx < len - 1 && symbols[read_idx] == left && symbols[read_idx + 1] == right {
            symbols[write_idx] = merged.to_string();
            write_idx += 1;
            read_idx += 2;
            changed = true;
        } else {
            symbols[write_idx] = symbols[read_idx].clone();
            write_idx += 1;
            read_idx += 1;
        }
    }
    if changed {
        symbols.truncate(write_idx);
    }
    changed
}

fn has_long_corpus_entry(corpus: &[WordSymbols]) -> bool {
    for entry in corpus {
        if entry.symbols.len() > 3 {
            return true;
        }
    }
    false
}

impl AdaptiveTokenizer {
    pub fn known_token_count(&self) -> usize {
        self.entries.len()
    }

    pub fn contains(&self, surface: &str) -> bool {
        self.lookup.contains_key(surface)
    }

    pub fn get(&self, node_id: u64) -> Option<&TokenEntry> {
        self.entries.get(&node_id)
    }

    pub fn record_surface(&mut self, surface: &str) -> u64 {
        let count = self.surface_counts.entry(surface.to_string()).or_insert(0);
        *count += 1;
        *count
    }

    pub fn register_token(
        &mut self,
        node_id: u64,
        surface: &str,
        level: TokenLevel,
        interaction_index: u64,
    ) {
        let key = match level {
            TokenLevel::Sensor => surface.to_string(),
            TokenLevel::Word | TokenLevel::Phrase => {
                if surface.starts_with(WORD_BOUNDARY) {
                    surface.to_string()
                } else {
                    format!("{}{}", WORD_BOUNDARY, surface)
                }
            }
        };
        self.lookup.insert(key.clone(), node_id);
        self.entries.insert(
            node_id,
            TokenEntry {
                node_id,
                text: key.clone(),
                level,
                occurrence_count: 0,
                created_at: interaction_index,
                last_used_at: interaction_index,
            },
        );
        self.trie.insert(&key, node_id, level);
    }

    pub fn touch_token(&mut self, node_id: u64, interaction_index: u64) -> Result<(), BrainError> {
        let token = self
            .entries
            .get_mut(&node_id)
            .ok_or(BrainError::MissingToken(node_id))?;
        token.occurrence_count += 1;
        token.last_used_at = interaction_index;
        Ok(())
    }

    pub fn rebuild_trie(&mut self) {
        self.trie = TokenTrie::default();
        for entry in self.entries.values() {
            self.trie.insert(&entry.text, entry.node_id, entry.level);
        }
    }

    pub fn tokenize(&self, text: &str) -> Vec<u64> {
        let words = pre_tokenize(text);
        let mut token_ids = Vec::new();

        for word in words {
            if word.is_empty() {
                continue;
            }
            let full_word = format!("{}{}", WORD_BOUNDARY, word);

            // 1. Whole-word lookup
            if let Some(&id) = self.lookup.get(&full_word) {
                token_ids.push(id);
                continue;
            }

            // 2. Character decomposition & merge rules application
            let mut symbols = create_initial_symbols(&full_word);
            for (left, right) in &self.merges {
                let merged = format!("{}{}", left, right);
                apply_merge_in_place(&mut symbols, left, right, &merged);
            }

            // 3. Map symbols to IDs
            for sym in symbols {
                if let Some(&id) = self.lookup.get(&sym) {
                    token_ids.push(id);
                } else {
                    if let Some(&unk_id) = self.lookup.get(UNK_TOKEN) {
                        token_ids.push(unk_id);
                    }
                }
            }
        }

        token_ids
    }

    pub fn decode_tokens(&self, token_ids: &[u64]) -> Result<String, BrainError> {
        let mut tokens = Vec::new();
        for &id in token_ids {
            if let Some(entry) = self.get(id) {
                let text = &entry.text;
                if text != BOS_TOKEN && text != EOS_TOKEN && text != PAD_TOKEN && text != UNK_TOKEN {
                    tokens.push(text.clone());
                }
            } else {
                return Err(BrainError::MissingToken(id));
            }
        }
        let merged = tokens.join("").replace(WORD_BOUNDARY, " ");
        Ok(collapse_whitespace(&merged))
    }

    pub fn train(&mut self, texts: &[String]) {
        self.lookup.clear();
        self.merges.clear();
        self.entries.clear();
        self.surface_counts.clear();

        let mut word_freq: BTreeMap<String, usize> = BTreeMap::new();
        for text in texts {
            let words = pre_tokenize(text);
            for word in words {
                if word.is_empty() {
                    continue;
                }
                let key = format!("{}{}", WORD_BOUNDARY, word);
                *word_freq.entry(key).or_insert(0) += 1;
            }
        }

        let mut corpus: Vec<WordSymbols> = Vec::new();
        let mut next_id = 0;

        let special_tokens = vec![
            PAD_TOKEN.to_string(),
            UNK_TOKEN.to_string(),
            BOS_TOKEN.to_string(),
            EOS_TOKEN.to_string(),
        ];
        for tok in &special_tokens {
            self.lookup.insert(tok.clone(), next_id);
            next_id += 1;
        }

        for (word, freq) in word_freq {
            let chars = create_initial_symbols(&word);
            for char_sym in &chars {
                if !self.lookup.contains_key(char_sym) {
                    self.lookup.insert(char_sym.clone(), next_id);
                    next_id += 1;
                }
            }
            corpus.push(WordSymbols { symbols: chars, freq });
        }

        while self.lookup.len() < self.vocab_size || has_long_corpus_entry(&corpus) {
            let mut pair_freq: BTreeMap<(String, String), usize> = BTreeMap::new();
            for entry in &corpus {
                for i in 0..entry.symbols.len().saturating_sub(1) {
                    let left = &entry.symbols[i];
                    let right = &entry.symbols[i + 1];
                    let merged = format!("{}{}", left, right);
                    if !is_mergeable_token(&merged, &special_tokens) {
                        continue;
                    }
                    pair_freq.entry((left.clone(), right.clone()))
                        .and_modify(|f| *f += entry.freq)
                        .or_insert(entry.freq);
                }
            }

            if pair_freq.is_empty() {
                break;
            }

            let mut best_pair = None;
            let mut best_freq = 0;
            for (pair, freq) in pair_freq {
                if freq > best_freq {
                    best_freq = freq;
                    best_pair = Some(pair);
                }
            }

            let (left, right) = match best_pair {
                Some(p) => p,
                None => break,
            };

            if best_freq < self.min_frequency {
                break;
            }

            let merged = format!("{}{}", left, right);
            let already_merged = self.merges.iter().any(|(l, r)| l == &left && r == &right);

            if !already_merged {
                if !self.lookup.contains_key(&merged) {
                    self.merges.push((left.clone(), right.clone()));
                    self.lookup.insert(merged.clone(), next_id);
                    next_id += 1;
                }

                for entry in &mut corpus {
                    apply_merge_in_place(&mut entry.symbols, &left, &right, &merged);
                }
            } else {
                break;
            }
        }

        let mut unused_idx = 0;
        while self.lookup.len() < self.vocab_size {
            let placeholder = format!("<UNUSED_{}>", unused_idx);
            unused_idx += 1;
            if !self.lookup.contains_key(&placeholder) {
                self.lookup.insert(placeholder.clone(), next_id);
                next_id += 1;
            }
        }

        self.build_entries(0);
    }

    pub fn build_entries(&mut self, interaction_index: u64) {
        self.entries.clear();
        for (surface, &node_id) in &self.lookup {
            let level = if surface == PAD_TOKEN || surface == UNK_TOKEN || surface == BOS_TOKEN || surface == EOS_TOKEN {
                TokenLevel::Sensor
            } else if surface.chars().count() <= 1 {
                TokenLevel::Sensor
            } else {
                let clean = surface.replace(WORD_BOUNDARY, "");
                if clean.contains(' ') {
                    TokenLevel::Phrase
                } else {
                    TokenLevel::Word
                }
            };
            self.entries.insert(
                node_id,
                TokenEntry {
                    node_id,
                    text: surface.clone(),
                    level,
                    occurrence_count: 0,
                    created_at: interaction_index,
                    last_used_at: interaction_index,
                },
            );
        }
        self.rebuild_trie();
    }
}

pub fn is_punctuation_or_space(character: char) -> bool {
    character.is_whitespace() || character.is_ascii_punctuation()
}

pub fn collapse_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub fn token_priority(level: TokenLevel) -> u8 {
    match level {
        TokenLevel::Sensor => 0,
        TokenLevel::Word => 1,
        TokenLevel::Phrase => 2,
    }
}
