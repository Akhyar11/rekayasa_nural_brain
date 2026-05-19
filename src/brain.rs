use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::error::Error;
use std::fmt;
use std::fs;
use std::io;
use std::path::Path;

const STATE_VERSION: u32 = 1;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BrainConfig {
    pub word_promotion_threshold: u64,
    pub phrase_promotion_threshold: u64,
    pub context_promotion_threshold: u64,
    pub max_ngram: usize,
    pub max_context_window: usize,
    pub prune_interval: u64,
    pub edge_decay: f32,
    pub min_edge_strength: f32,
    pub response_token_limit: usize,
    pub max_recent_utterances: usize,
}

impl Default for BrainConfig {
    fn default() -> Self {
        Self {
            word_promotion_threshold: 2,
            phrase_promotion_threshold: 3,
            context_promotion_threshold: 3,
            max_ngram: 3,
            max_context_window: 3,
            prune_interval: 24,
            edge_decay: 0.97,
            min_edge_strength: 0.12,
            response_token_limit: 24,
            max_recent_utterances: 128,
        }
    }
}

impl BrainConfig {
    pub fn validate(&self) -> Result<(), BrainError> {
        if self.word_promotion_threshold == 0 {
            return Err(BrainError::InvalidConfig(
                "word-promotion-threshold harus lebih besar dari 0",
            ));
        }
        if self.phrase_promotion_threshold == 0 {
            return Err(BrainError::InvalidConfig(
                "phrase-promotion-threshold harus lebih besar dari 0",
            ));
        }
        if self.context_promotion_threshold == 0 {
            return Err(BrainError::InvalidConfig(
                "context-promotion-threshold harus lebih besar dari 0",
            ));
        }
        if self.max_ngram < 2 {
            return Err(BrainError::InvalidConfig(
                "max-ngram minimal 2 agar pembentukan frasa bermakna",
            ));
        }
        if self.max_context_window == 0 {
            return Err(BrainError::InvalidConfig(
                "max-context-window harus lebih besar dari 0",
            ));
        }
        if self.prune_interval == 0 {
            return Err(BrainError::InvalidConfig(
                "prune-interval harus lebih besar dari 0",
            ));
        }
        if !self.edge_decay.is_finite() || !(0.0..=1.0).contains(&self.edge_decay) {
            return Err(BrainError::InvalidConfig(
                "edge-decay harus bernilai finite di antara 0.0 dan 1.0",
            ));
        }
        if !self.min_edge_strength.is_finite() || self.min_edge_strength < 0.0 {
            return Err(BrainError::InvalidConfig(
                "min-edge-strength harus bernilai finite dan >= 0",
            ));
        }
        if self.response_token_limit == 0 {
            return Err(BrainError::InvalidConfig(
                "response-token-limit harus lebih besar dari 0",
            ));
        }
        if self.max_recent_utterances == 0 {
            return Err(BrainError::InvalidConfig(
                "max-recent-utterances harus lebih besar dari 0",
            ));
        }

        Ok(())
    }
}

#[derive(Debug)]
pub enum BrainError {
    EmptyInput,
    InvalidConfig(&'static str),
    Io(io::Error),
    Encode(bincode::error::EncodeError),
    Decode(bincode::error::DecodeError),
    MissingToken(u64),
    MissingNode(u64),
    UnsupportedStateVersion(u32),
}

impl fmt::Display for BrainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "input tidak boleh kosong"),
            Self::InvalidConfig(message) => write!(f, "{message}"),
            Self::Io(error) => write!(f, "{error}"),
            Self::Encode(error) => write!(f, "{error}"),
            Self::Decode(error) => write!(f, "{error}"),
            Self::MissingToken(node_id) => {
                write!(f, "token untuk node #{node_id} tidak ditemukan")
            }
            Self::MissingNode(node_id) => write!(f, "node #{node_id} tidak ditemukan"),
            Self::UnsupportedStateVersion(version) => {
                write!(f, "versi state {version} tidak didukung")
            }
        }
    }
}

impl Error for BrainError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Encode(error) => Some(error),
            Self::Decode(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for BrainError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<bincode::error::EncodeError> for BrainError {
    fn from(error: bincode::error::EncodeError) -> Self {
        Self::Encode(error)
    }
}

impl From<bincode::error::DecodeError> for BrainError {
    fn from(error: bincode::error::DecodeError) -> Self {
        Self::Decode(error)
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum TokenLevel {
    Sensor,
    Word,
    Phrase,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum NodeKind {
    Sensor,
    Lexical,
    Phrase,
    Context,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum EdgeKind {
    Transition,
    ContextInput,
    ContextPrediction,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BrainNode {
    pub id: u64,
    pub label: String,
    pub kind: NodeKind,
    pub activation_count: u64,
    pub last_activated_at: u64,
    pub salience: f32,
    pub composition: Vec<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BrainEdge {
    pub source: u64,
    pub target: u64,
    pub kind: EdgeKind,
    pub strength: f32,
    pub activation_count: u64,
    pub last_activated_at: u64,
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

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AdaptiveTokenizer {
    pub lookup: BTreeMap<String, u64>,
    pub entries: BTreeMap<u64, TokenEntry>,
    pub surface_counts: BTreeMap<String, u64>,
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

    pub fn tokenize(&self, normalized_text: &str) -> Vec<u64> {
        let chars: Vec<char> = normalized_text.chars().collect();
        let candidate_ids = self.sorted_candidate_ids();
        let mut token_ids = Vec::new();
        let mut cursor = 0;

        while cursor < chars.len() {
            let mut matched = None;
            for node_id in &candidate_ids {
                let Some(token) = self.entries.get(node_id) else {
                    continue;
                };
                if token.text.is_empty() {
                    continue;
                }
                if token_matches_at(token, &chars, cursor) {
                    matched = Some((*node_id, token.text.chars().count()));
                    break;
                }
            }

            if let Some((node_id, width)) = matched {
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

    fn sorted_candidate_ids(&self) -> Vec<u64> {
        let mut token_ids: Vec<u64> = self.entries.keys().copied().collect();
        token_ids.sort_by(|left, right| {
            let left_token = self
                .entries
                .get(left)
                .expect("sorted_candidate_ids should only use existing ids");
            let right_token = self
                .entries
                .get(right)
                .expect("sorted_candidate_ids should only use existing ids");

            right_token
                .text
                .chars()
                .count()
                .cmp(&left_token.text.chars().count())
                .then_with(|| {
                    token_priority(right_token.level).cmp(&token_priority(left_token.level))
                })
                .then_with(|| left.cmp(right))
        });
        token_ids
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContextPattern {
    pub token_ids: Vec<u64>,
    pub occurrence_count: u64,
    pub predicted_counts: BTreeMap<u64, u64>,
    pub node_id: Option<u64>,
    pub last_activated_at: u64,
    pub label: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UtteranceMemory {
    pub interaction_index: u64,
    pub text: String,
    pub token_ids: Vec<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BrainState {
    pub state_version: u32,
    pub config: BrainConfig,
    pub next_node_id: u64,
    pub interaction_count: u64,
    pub tokenizer: AdaptiveTokenizer,
    pub nodes: BTreeMap<u64, BrainNode>,
    pub edges: BTreeMap<String, BrainEdge>,
    pub context_patterns: BTreeMap<String, ContextPattern>,
    pub recent_utterances: VecDeque<UtteranceMemory>,
}

impl BrainState {
    pub fn new(config: BrainConfig) -> Result<Self, BrainError> {
        config.validate()?;
        Ok(Self {
            state_version: STATE_VERSION,
            config,
            next_node_id: 1,
            interaction_count: 0,
            tokenizer: AdaptiveTokenizer::default(),
            nodes: BTreeMap::new(),
            edges: BTreeMap::new(),
            context_patterns: BTreeMap::new(),
            recent_utterances: VecDeque::new(),
        })
    }

    pub fn load_or_new(path: &Path, config: BrainConfig) -> Result<Self, BrainError> {
        if path.exists() {
            let mut state = Self::load_from_path(path)?;
            config.validate()?;
            state.config = config;
            Ok(state)
        } else {
            Self::new(config)
        }
    }

    pub fn load_from_path(path: &Path) -> Result<Self, BrainError> {
        let bytes = fs::read(path)?;
        let (state, _bytes_read): (Self, usize) =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard())?;
        if state.state_version != STATE_VERSION {
            return Err(BrainError::UnsupportedStateVersion(state.state_version));
        }
        state.config.validate()?;
        Ok(state)
    }

    pub fn save_to_path(&self, path: &Path) -> Result<(), BrainError> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }
        let payload = bincode::serde::encode_to_vec(self, bincode::config::standard())?;
        fs::write(path, payload)?;
        Ok(())
    }

    pub fn interact(&mut self, input: &str) -> Result<InteractionReport, BrainError> {
        let learning = self.learn_text(input)?;
        let response = self.generate_response_from_tokens(&learning.token_ids)?;
        Ok(InteractionReport { learning, response })
    }

    pub fn learn_text(&mut self, input: &str) -> Result<LearningReport, BrainError> {
        self.config.validate()?;
        let normalized_text = normalize_input(input)?;
        self.interaction_count += 1;
        let interaction_index = self.interaction_count;
        let mut report = LearningReport {
            normalized_text: normalized_text.clone(),
            token_count: 0,
            new_sensor_tokens: Vec::new(),
            new_word_tokens: Vec::new(),
            new_phrase_tokens: Vec::new(),
            new_context_nodes: Vec::new(),
            new_edges: 0,
            pruned_edges: 0,
            pruned_context_nodes: 0,
            token_ids: Vec::new(),
        };

        self.ensure_sensor_tokens(&normalized_text, interaction_index, &mut report)?;
        self.promote_word_tokens(&normalized_text, interaction_index, &mut report)?;
        self.promote_phrase_tokens(&normalized_text, interaction_index, &mut report)?;

        let token_ids = self.tokenizer.tokenize(&normalized_text);
        report.token_count = token_ids.len();
        report.token_ids = token_ids.clone();
        if token_ids.is_empty() {
            return Err(BrainError::EmptyInput);
        }

        for token_id in &token_ids {
            self.activate_node(*token_id, interaction_index)?;
            self.tokenizer.touch_token(*token_id, interaction_index)?;
        }

        report.new_edges += self.learn_transitions(&token_ids, interaction_index);
        report.new_edges +=
            self.learn_context_patterns(&token_ids, interaction_index, &mut report)?;
        self.push_utterance(interaction_index, &normalized_text, token_ids);

        if interaction_index.is_multiple_of(self.config.prune_interval) {
            let (pruned_edges, pruned_nodes) = self.prune_graph(interaction_index);
            report.pruned_edges = pruned_edges;
            report.pruned_context_nodes = pruned_nodes;
        }

        Ok(report)
    }

    pub fn generate_response(&self, prompt: &str) -> Result<String, BrainError> {
        let normalized_text = normalize_input(prompt)?;
        let token_ids = self.tokenizer.tokenize(&normalized_text);
        self.generate_response_from_tokens(&token_ids)
    }

    pub fn summary(&self, strongest_edge_limit: usize) -> BrainSummary {
        let mut strongest_edges: Vec<&BrainEdge> = self.edges.values().collect();
        strongest_edges.sort_by(|left, right| {
            right
                .strength
                .total_cmp(&left.strength)
                .then_with(|| right.activation_count.cmp(&left.activation_count))
        });

        let strongest_edges = strongest_edges
            .into_iter()
            .take(strongest_edge_limit)
            .map(|edge| BrainEdgeSummary {
                source_label: self.node_label(edge.source),
                target_label: self.node_label(edge.target),
                kind: edge.kind,
                strength: edge.strength,
                activation_count: edge.activation_count,
            })
            .collect();

        let mut latest_tokens: Vec<&TokenEntry> = self.tokenizer.entries.values().collect();
        latest_tokens.sort_by(|left, right| {
            right
                .created_at
                .cmp(&left.created_at)
                .then_with(|| right.node_id.cmp(&left.node_id))
        });

        BrainSummary {
            interactions: self.interaction_count,
            token_count: self.tokenizer.entries.len(),
            sensor_token_count: self
                .tokenizer
                .entries
                .values()
                .filter(|token| token.level == TokenLevel::Sensor)
                .count(),
            word_token_count: self
                .tokenizer
                .entries
                .values()
                .filter(|token| token.level == TokenLevel::Word)
                .count(),
            phrase_token_count: self
                .tokenizer
                .entries
                .values()
                .filter(|token| token.level == TokenLevel::Phrase)
                .count(),
            context_node_count: self
                .nodes
                .values()
                .filter(|node| node.kind == NodeKind::Context)
                .count(),
            edge_count: self.edges.len(),
            remembered_utterances: self.recent_utterances.len(),
            latest_tokens: latest_tokens
                .into_iter()
                .take(8)
                .map(|token| token.text.clone())
                .collect(),
            strongest_edges,
        }
    }

    fn ensure_sensor_tokens(
        &mut self,
        normalized_text: &str,
        interaction_index: u64,
        report: &mut LearningReport,
    ) -> Result<(), BrainError> {
        let mut observed = BTreeSet::new();
        for character in normalized_text.chars() {
            let surface = character.to_string();
            self.tokenizer.record_surface(&surface);
            if self.tokenizer.contains(&surface) {
                continue;
            }
            if observed.insert(surface.clone()) {
                self.create_token_node(&surface, TokenLevel::Sensor, interaction_index)?;
                report.new_sensor_tokens.push(surface);
            }
        }
        Ok(())
    }

    fn promote_word_tokens(
        &mut self,
        normalized_text: &str,
        interaction_index: u64,
        report: &mut LearningReport,
    ) -> Result<(), BrainError> {
        let mut promoted = BTreeSet::new();
        for word in normalized_text.split_whitespace() {
            let count = self.tokenizer.record_surface(word);
            let should_promote = count >= self.config.word_promotion_threshold
                && !self.tokenizer.contains(word)
                && word.chars().count() > 1;
            if should_promote && promoted.insert(word.to_string()) {
                self.create_token_node(word, TokenLevel::Word, interaction_index)?;
                report.new_word_tokens.push(word.to_string());
            }
        }
        Ok(())
    }

    fn promote_phrase_tokens(
        &mut self,
        normalized_text: &str,
        interaction_index: u64,
        report: &mut LearningReport,
    ) -> Result<(), BrainError> {
        let words: Vec<&str> = normalized_text.split_whitespace().collect();
        if words.len() < 2 {
            return Ok(());
        }

        let max_ngram = self.config.max_ngram.min(words.len());
        let mut promoted = BTreeSet::new();

        for ngram_size in 2..=max_ngram {
            for start in 0..=words.len() - ngram_size {
                let phrase = words[start..start + ngram_size].join(" ");
                let count = self.tokenizer.record_surface(&phrase);
                let should_promote = count >= self.config.phrase_promotion_threshold
                    && !self.tokenizer.contains(&phrase);
                if should_promote && promoted.insert(phrase.clone()) {
                    self.create_token_node(&phrase, TokenLevel::Phrase, interaction_index)?;
                    report.new_phrase_tokens.push(phrase);
                }
            }
        }
        Ok(())
    }

    fn create_token_node(
        &mut self,
        surface: &str,
        level: TokenLevel,
        interaction_index: u64,
    ) -> Result<u64, BrainError> {
        if let Some(existing) = self.tokenizer.lookup.get(surface).copied() {
            return Ok(existing);
        }

        let node_id = self.allocate_node_id();
        let kind = match level {
            TokenLevel::Sensor => NodeKind::Sensor,
            TokenLevel::Word => NodeKind::Lexical,
            TokenLevel::Phrase => NodeKind::Phrase,
        };
        let composition = match level {
            TokenLevel::Sensor => Vec::new(),
            TokenLevel::Word | TokenLevel::Phrase => self.tokenizer.tokenize(surface),
        };
        self.nodes.insert(
            node_id,
            BrainNode {
                id: node_id,
                label: surface.to_string(),
                kind,
                activation_count: 0,
                last_activated_at: interaction_index,
                salience: 0.0,
                composition,
            },
        );
        self.tokenizer
            .register_token(node_id, surface, level, interaction_index);
        Ok(node_id)
    }

    fn create_context_node(
        &mut self,
        token_ids: &[u64],
        interaction_index: u64,
    ) -> Result<u64, BrainError> {
        let node_id = self.allocate_node_id();
        let label = format!("ctx: {}", self.describe_tokens(token_ids));
        self.nodes.insert(
            node_id,
            BrainNode {
                id: node_id,
                label,
                kind: NodeKind::Context,
                activation_count: 0,
                last_activated_at: interaction_index,
                salience: 0.0,
                composition: token_ids.to_vec(),
            },
        );
        Ok(node_id)
    }

    fn learn_transitions(&mut self, token_ids: &[u64], interaction_index: u64) -> usize {
        let mut updates = 0;
        for window in token_ids.windows(2) {
            if let [source, target] = window {
                self.strengthen_edge(
                    *source,
                    *target,
                    EdgeKind::Transition,
                    interaction_index,
                    1.0,
                );
                updates += 1;
            }
        }
        updates
    }

    fn learn_context_patterns(
        &mut self,
        token_ids: &[u64],
        interaction_index: u64,
        report: &mut LearningReport,
    ) -> Result<usize, BrainError> {
        if token_ids.len() < 2 {
            return Ok(0);
        }

        let mut updates = 0;
        let mut pending_links = Vec::new();
        let max_window = self.config.max_context_window.min(token_ids.len() - 1);

        for window_size in 1..=max_window {
            for start in 0..=token_ids.len() - window_size - 1 {
                let prefix = token_ids[start..start + window_size].to_vec();
                let predicted = token_ids[start + window_size];
                let key = context_key(&prefix);
                let label = self.describe_tokens(&prefix);

                let mut should_promote = false;
                {
                    let pattern = self.context_patterns.entry(key.clone()).or_insert_with(|| {
                        ContextPattern {
                            token_ids: prefix.clone(),
                            occurrence_count: 0,
                            predicted_counts: BTreeMap::new(),
                            node_id: None,
                            last_activated_at: interaction_index,
                            label: label.clone(),
                        }
                    });
                    pattern.occurrence_count += 1;
                    *pattern.predicted_counts.entry(predicted).or_insert(0) += 1;
                    pattern.last_activated_at = interaction_index;
                    if pattern.node_id.is_none()
                        && pattern.occurrence_count >= self.config.context_promotion_threshold
                    {
                        should_promote = true;
                    }
                }

                if should_promote {
                    let node_id = self.create_context_node(&prefix, interaction_index)?;
                    let pattern = self
                        .context_patterns
                        .get_mut(&key)
                        .expect("context pattern should exist after insertion");
                    pattern.node_id = Some(node_id);
                    report.new_context_nodes.push(pattern.label.clone());
                }

                if let Some(node_id) = self
                    .context_patterns
                    .get(&key)
                    .and_then(|pattern| pattern.node_id)
                {
                    pending_links.push((prefix, node_id, predicted));
                }
            }
        }

        for (prefix, node_id, predicted) in pending_links {
            self.activate_node(node_id, interaction_index)?;
            for source in &prefix {
                self.strengthen_edge(
                    *source,
                    node_id,
                    EdgeKind::ContextInput,
                    interaction_index,
                    0.35,
                );
                updates += 1;
            }
            self.strengthen_edge(
                node_id,
                predicted,
                EdgeKind::ContextPrediction,
                interaction_index,
                0.8,
            );
            updates += 1;
        }

        Ok(updates)
    }

    fn push_utterance(&mut self, interaction_index: u64, text: &str, token_ids: Vec<u64>) {
        if self.recent_utterances.len() >= self.config.max_recent_utterances {
            self.recent_utterances.pop_front();
        }
        self.recent_utterances.push_back(UtteranceMemory {
            interaction_index,
            text: text.to_string(),
            token_ids,
        });
    }

    fn generate_response_from_tokens(
        &self,
        prompt_token_ids: &[u64],
    ) -> Result<String, BrainError> {
        if prompt_token_ids.is_empty() {
            return Ok("saya belum punya cukup pola untuk merespons.".to_string());
        }

        let mut generated = Vec::new();
        let mut context = self.expand_context_tokens(prompt_token_ids);
        if context.is_empty() {
            context = prompt_token_ids.to_vec();
        }
        let mut seen = BTreeSet::new();

        for _ in 0..self.config.response_token_limit {
            let Some((next_token, confidence)) = self.predict_next_token(&context) else {
                break;
            };
            if confidence < 0.34 {
                break;
            }
            if !seen.insert((
                context_key(
                    &context[context.len().saturating_sub(self.config.max_context_window)..],
                ),
                next_token,
            )) {
                break;
            }
            generated.push(next_token);
            context.push(next_token);
            if let Some(token) = self.tokenizer.get(next_token)
                && token.level != TokenLevel::Sensor
                && token.text.ends_with(['.', '!', '?'])
            {
                break;
            }
        }

        if generated.is_empty() {
            if let Some(recalled) = self.recent_utterances.back() {
                return Ok(format!("saya menyimpan pola: {}", recalled.text));
            }
            return Ok("saya masih membangun pola dari interaksi pertama.".to_string());
        }

        let decoded = self.tokenizer.decode_tokens(&generated)?;
        if decoded.is_empty() {
            Ok("saya mengenali inputnya, tetapi belum punya kelanjutan yang stabil.".to_string())
        } else {
            Ok(decoded)
        }
    }

    fn expand_context_tokens(&self, token_ids: &[u64]) -> Vec<u64> {
        let mut expanded = Vec::new();
        for token_id in token_ids {
            self.expand_token_into(*token_id, &mut expanded, 0);
        }
        expanded
    }

    fn expand_token_into(&self, token_id: u64, output: &mut Vec<u64>, depth: usize) {
        if depth >= 4 {
            output.push(token_id);
            return;
        }

        let Some(node) = self.nodes.get(&token_id) else {
            output.push(token_id);
            return;
        };

        if node.composition.is_empty() {
            output.push(token_id);
            return;
        }

        for child in &node.composition {
            self.expand_token_into(*child, output, depth + 1);
        }
    }

    fn predict_next_token(&self, context: &[u64]) -> Option<(u64, f32)> {
        let max_window = self.config.max_context_window.min(context.len());
        for window_size in (1..=max_window).rev() {
            let prefix = &context[context.len() - window_size..];
            if let Some(pattern) = self.context_patterns.get(&context_key(prefix)) {
                let total_predictions: u64 = pattern.predicted_counts.values().sum();
                if total_predictions == 0 {
                    continue;
                }
                if let Some((token_id, count)) = pattern.predicted_counts.iter().max_by(
                    |(left_id, left_count), (right_id, right_count)| {
                        left_count.cmp(right_count).then_with(|| {
                            token_priority(
                                self.tokenizer
                                    .get(**left_id)
                                    .map(|token| token.level)
                                    .unwrap_or(TokenLevel::Sensor),
                            )
                            .cmp(&token_priority(
                                self.tokenizer
                                    .get(**right_id)
                                    .map(|token| token.level)
                                    .unwrap_or(TokenLevel::Sensor),
                            ))
                        })
                    },
                ) {
                    return Some((*token_id, *count as f32 / total_predictions as f32));
                }
            }
        }

        let last_token = *context.last()?;
        let mut best_edge: Option<&BrainEdge> = None;
        for edge in self.edges.values() {
            if edge.kind != EdgeKind::Transition || edge.source != last_token {
                continue;
            }
            best_edge = match best_edge {
                Some(current) => {
                    if edge.strength > current.strength {
                        Some(edge)
                    } else {
                        Some(current)
                    }
                }
                None => Some(edge),
            };
        }

        best_edge.map(|edge| (edge.target, edge.strength.clamp(0.0, 1.0)))
    }

    fn prune_graph(&mut self, interaction_index: u64) -> (usize, usize) {
        let stale_after = self.config.prune_interval.saturating_mul(4);
        let mut edges_to_remove = Vec::new();
        for (key, edge) in &mut self.edges {
            let age = interaction_index.saturating_sub(edge.last_activated_at);
            if age > 1 {
                edge.strength *= self.config.edge_decay;
            }
            if edge.strength < self.config.min_edge_strength
                && age >= stale_after
                && edge.activation_count <= 1
            {
                edges_to_remove.push(key.clone());
            }
        }

        for key in &edges_to_remove {
            self.edges.remove(key);
        }

        let active_nodes: BTreeSet<u64> = self
            .edges
            .values()
            .flat_map(|edge| [edge.source, edge.target])
            .collect();
        let protected_token_nodes: BTreeSet<u64> = self.tokenizer.entries.keys().copied().collect();
        let mut context_nodes_to_remove = Vec::new();

        for (node_id, node) in &self.nodes {
            if node.kind != NodeKind::Context {
                continue;
            }
            if protected_token_nodes.contains(node_id) {
                continue;
            }
            let age = interaction_index.saturating_sub(node.last_activated_at);
            if !active_nodes.contains(node_id) && age >= stale_after {
                context_nodes_to_remove.push(*node_id);
            }
        }

        for node_id in &context_nodes_to_remove {
            self.nodes.remove(node_id);
        }

        for pattern in self.context_patterns.values_mut() {
            if let Some(node_id) = pattern.node_id
                && context_nodes_to_remove.contains(&node_id)
            {
                pattern.node_id = None;
            }
        }

        (edges_to_remove.len(), context_nodes_to_remove.len())
    }

    fn strengthen_edge(
        &mut self,
        source: u64,
        target: u64,
        kind: EdgeKind,
        interaction_index: u64,
        amount: f32,
    ) {
        let key = edge_key(source, target, kind);
        let edge = self.edges.entry(key).or_insert(BrainEdge {
            source,
            target,
            kind,
            strength: 0.0,
            activation_count: 0,
            last_activated_at: interaction_index,
        });
        edge.strength = (edge.strength + amount).clamp(0.0, 1.0);
        edge.activation_count += 1;
        edge.last_activated_at = interaction_index;
    }

    fn activate_node(&mut self, node_id: u64, interaction_index: u64) -> Result<(), BrainError> {
        let node = self
            .nodes
            .get_mut(&node_id)
            .ok_or(BrainError::MissingNode(node_id))?;
        node.activation_count += 1;
        node.last_activated_at = interaction_index;
        node.salience = (node.salience + 0.2).clamp(0.0, 1.0);
        Ok(())
    }

    fn allocate_node_id(&mut self) -> u64 {
        let node_id = self.next_node_id;
        self.next_node_id += 1;
        node_id
    }

    fn describe_tokens(&self, token_ids: &[u64]) -> String {
        let surfaces: Vec<String> = token_ids
            .iter()
            .filter_map(|token_id| self.tokenizer.get(*token_id))
            .map(|token| token.text.clone())
            .collect();
        collapse_whitespace(&surfaces.join(" "))
    }

    fn node_label(&self, node_id: u64) -> String {
        self.nodes
            .get(&node_id)
            .map(|node| node.label.clone())
            .unwrap_or_else(|| format!("node#{node_id}"))
    }
}

#[derive(Clone, Debug)]
pub struct LearningReport {
    pub normalized_text: String,
    pub token_count: usize,
    pub new_sensor_tokens: Vec<String>,
    pub new_word_tokens: Vec<String>,
    pub new_phrase_tokens: Vec<String>,
    pub new_context_nodes: Vec<String>,
    pub new_edges: usize,
    pub pruned_edges: usize,
    pub pruned_context_nodes: usize,
    pub token_ids: Vec<u64>,
}

#[derive(Clone, Debug)]
pub struct InteractionReport {
    pub learning: LearningReport,
    pub response: String,
}

#[derive(Clone, Debug)]
pub struct BrainEdgeSummary {
    pub source_label: String,
    pub target_label: String,
    pub kind: EdgeKind,
    pub strength: f32,
    pub activation_count: u64,
}

#[derive(Clone, Debug)]
pub struct BrainSummary {
    pub interactions: u64,
    pub token_count: usize,
    pub sensor_token_count: usize,
    pub word_token_count: usize,
    pub phrase_token_count: usize,
    pub context_node_count: usize,
    pub edge_count: usize,
    pub remembered_utterances: usize,
    pub latest_tokens: Vec<String>,
    pub strongest_edges: Vec<BrainEdgeSummary>,
}

fn normalize_input(text: &str) -> Result<String, BrainError> {
    let mut output = String::new();
    let mut previous_was_space = true;

    for character in text.chars() {
        if character.is_control() && !character.is_whitespace() {
            continue;
        }
        if character.is_whitespace() {
            if !previous_was_space && !output.is_empty() {
                output.push(' ');
            }
            previous_was_space = true;
        } else {
            previous_was_space = false;
            for lowered in character.to_lowercase() {
                output.push(lowered);
            }
        }
    }

    let collapsed = collapse_whitespace(&output);
    if collapsed.is_empty() {
        Err(BrainError::EmptyInput)
    } else {
        Ok(collapsed)
    }
}

fn collapse_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn token_priority(level: TokenLevel) -> u8 {
    match level {
        TokenLevel::Sensor => 0,
        TokenLevel::Word => 1,
        TokenLevel::Phrase => 2,
    }
}

fn token_matches_at(token: &TokenEntry, chars: &[char], start: usize) -> bool {
    let token_chars: Vec<char> = token.text.chars().collect();
    let end = start + token_chars.len();
    if end > chars.len() {
        return false;
    }
    if chars[start..end] != token_chars[..] {
        return false;
    }
    if token.level == TokenLevel::Sensor {
        return true;
    }

    let left_boundary_ok = start == 0 || !chars[start - 1].is_alphanumeric();
    let right_boundary_ok = end == chars.len() || !chars[end].is_alphanumeric();
    left_boundary_ok && right_boundary_ok
}

fn context_key(token_ids: &[u64]) -> String {
    token_ids
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(":")
}

fn edge_key(source: u64, target: u64, kind: EdgeKind) -> String {
    format!("{source}:{target}:{kind:?}")
}

fn is_punctuation_or_space(character: char) -> bool {
    character.is_whitespace() || character.is_ascii_punctuation()
}

#[cfg(test)]
mod tests {
    use super::{BrainConfig, BrainState, TokenLevel, normalize_input};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn normalize_input_collapses_whitespace() {
        let normalized =
            normalize_input("  Halo\tDunia \n Besar ").expect("normalization should pass");
        assert_eq!(normalized, "halo dunia besar");
    }

    #[test]
    fn brain_grows_word_and_phrase_tokens() {
        let mut brain = BrainState::new(BrainConfig::default()).expect("brain should initialize");

        brain
            .learn_text("halo dunia")
            .expect("learning should pass");
        let second = brain
            .learn_text("halo dunia")
            .expect("learning should pass");
        let third = brain
            .learn_text("halo dunia")
            .expect("learning should pass");

        assert!(second.new_word_tokens.iter().any(|token| token == "halo"));
        assert!(
            third
                .new_phrase_tokens
                .iter()
                .any(|token| token == "halo dunia")
        );
        assert!(
            brain
                .tokenizer
                .entries
                .values()
                .any(|token| token.level == TokenLevel::Phrase)
        );
    }

    #[test]
    fn brain_persists_and_loads_state() {
        let mut brain = BrainState::new(BrainConfig::default()).expect("brain should initialize");
        brain
            .learn_text("spiking brain")
            .expect("learning should pass");

        let temp_path = std::env::temp_dir().join(format!(
            "rekayasa_nural_brain_test_{}.bin",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock should be valid")
                .as_nanos()
        ));

        brain.save_to_path(&temp_path).expect("save should pass");
        let loaded = BrainState::load_from_path(&temp_path).expect("load should pass");
        std::fs::remove_file(&temp_path).expect("temp file should be removable");

        assert_eq!(loaded.interaction_count, 1);
        assert_eq!(
            loaded.tokenizer.known_token_count(),
            brain.tokenizer.known_token_count()
        );
        assert_eq!(loaded.recent_utterances.len(), 1);
    }

    #[test]
    fn interact_returns_response_after_learning() {
        let mut brain = BrainState::new(BrainConfig::default()).expect("brain should initialize");
        brain
            .learn_text("saya suka kopi")
            .expect("learning should pass");
        brain
            .learn_text("saya suka teh")
            .expect("learning should pass");
        let interaction = brain
            .interact("saya suka")
            .expect("interaction should pass");

        assert!(!interaction.response.is_empty());
        assert!(interaction.learning.token_count > 0);
    }

    #[test]
    fn summary_reports_graph_shape() {
        let mut brain = BrainState::new(BrainConfig::default()).expect("brain should initialize");
        brain
            .learn_text("otak ini tumbuh")
            .expect("learning should pass");
        let summary = brain.summary(4);

        assert!(summary.token_count > 0);
        assert!(summary.edge_count > 0);
    }
}
