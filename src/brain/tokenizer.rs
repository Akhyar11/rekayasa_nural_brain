use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use super::error::BrainError;

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

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AdaptiveTokenizer {
    pub lookup: BTreeMap<String, u64>,
    pub entries: BTreeMap<u64, TokenEntry>,
    pub surface_counts: BTreeMap<String, u64>,

    #[serde(skip, default)]
    pub trie: TokenTrie,
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
        self.lookup.insert(surface.to_string(), node_id);
        self.trie.insert(surface, node_id, level);
        self.entries.insert(
            node_id,
            TokenEntry {
                node_id,
                text: surface.to_string(),
                level,
                occurrence_count: 0,
                created_at: interaction_index,
                last_used_at: interaction_index,
            },
        );
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

    pub fn tokenize(&self, normalized_text: &str) -> Vec<u64> {
        let chars: Vec<char> = normalized_text.chars().collect();
        let mut token_ids = Vec::new();
        let mut cursor = 0;

        while cursor < chars.len() {
            let mut best_match: Option<(u64, usize)> = None;
            let mut current = &self.trie.root;
            let mut depth = 0;

            for &c in &chars[cursor..] {
                if let Some(next_node) = current.children.get(&c) {
                    current = next_node;
                    depth += 1;
                    if let Some(token_id) = current.token_id {
                        let level = current.level.expect("token should have level");
                        // check boundary condition
                        let end = cursor + depth;
                        let matched = if level == TokenLevel::Sensor {
                            true
                        } else {
                            let left_boundary_ok = cursor == 0 || !chars[cursor - 1].is_alphanumeric();
                            let right_boundary_ok = end == chars.len() || !chars[end].is_alphanumeric();
                            left_boundary_ok && right_boundary_ok
                        };
                        if matched {
                            best_match = Some((token_id, depth));
                        }
                    }
                } else {
                    break;
                }
            }

            if let Some((node_id, width)) = best_match {
                token_ids.push(node_id);
                cursor += width;
            } else {
                cursor += 1;
            }
        }

        token_ids
    }

    pub fn decode_tokens(&self, token_ids: &[u64]) -> Result<String, BrainError> {
        let mut output = String::new();

        for node_id in token_ids {
            let token = self
                .get(*node_id)
                .ok_or(BrainError::MissingToken(*node_id))?;
            match token.level {
                TokenLevel::Sensor => output.push_str(&token.text),
                TokenLevel::Word | TokenLevel::Phrase => {
                    if !output.is_empty()
                        && !output.ends_with(' ')
                        && !token.text.starts_with(is_punctuation_or_space)
                    {
                        output.push(' ');
                    }
                    output.push_str(&token.text);
                }
            }
        }

        Ok(collapse_whitespace(&output))
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
