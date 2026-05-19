use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};
use std::error::Error;
use std::fmt;
use std::io;
use std::path::PathBuf;

use super::tokenizer::AdaptiveTokenizer;

pub const STATE_VERSION: u32 = 3;
pub const LEGACY_STATE_VERSION_V2: u32 = 2;
pub const LEGACY_STATE_VERSION_V1: u32 = 1;

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
    SerdeJson(serde_json::Error),
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
            Self::SerdeJson(error) => write!(f, "{error}"),
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
            Self::SerdeJson(error) => Some(error),
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

impl From<serde_json::Error> for BrainError {
    fn from(error: serde_json::Error) -> Self {
        Self::SerdeJson(error)
    }
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
    pub learning_step_count: u64,
    pub training_example_count: u64,
    pub tokenizer: AdaptiveTokenizer,
    pub nodes: BTreeMap<u64, BrainNode>,
    pub edges: BTreeMap<String, BrainEdge>,
    pub context_patterns: BTreeMap<String, ContextPattern>,
    pub prompt_response_memory: BTreeMap<String, BTreeMap<String, u64>>,
    pub recent_utterances: VecDeque<UtteranceMemory>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LegacyBrainStateV2 {
    pub state_version: u32,
    pub config: BrainConfig,
    pub next_node_id: u64,
    pub interaction_count: u64,
    pub tokenizer: AdaptiveTokenizer,
    pub nodes: BTreeMap<u64, BrainNode>,
    pub edges: BTreeMap<String, BrainEdge>,
    pub context_patterns: BTreeMap<String, ContextPattern>,
    pub prompt_response_memory: BTreeMap<String, BTreeMap<String, u64>>,
    pub recent_utterances: VecDeque<UtteranceMemory>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LegacyBrainStateV1 {
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

impl From<LegacyBrainStateV2> for BrainState {
    fn from(legacy: LegacyBrainStateV2) -> Self {
        Self {
            state_version: STATE_VERSION,
            config: legacy.config,
            next_node_id: legacy.next_node_id,
            interaction_count: legacy.interaction_count,
            learning_step_count: legacy.interaction_count,
            training_example_count: 0,
            tokenizer: legacy.tokenizer,
            nodes: legacy.nodes,
            edges: legacy.edges,
            context_patterns: legacy.context_patterns,
            prompt_response_memory: legacy.prompt_response_memory,
            recent_utterances: legacy.recent_utterances,
        }
    }
}

impl From<LegacyBrainStateV1> for BrainState {
    fn from(legacy: LegacyBrainStateV1) -> Self {
        Self {
            state_version: STATE_VERSION,
            config: legacy.config,
            next_node_id: legacy.next_node_id,
            interaction_count: legacy.interaction_count,
            learning_step_count: legacy.interaction_count,
            training_example_count: 0,
            tokenizer: legacy.tokenizer,
            nodes: legacy.nodes,
            edges: legacy.edges,
            context_patterns: legacy.context_patterns,
            prompt_response_memory: BTreeMap::new(),
            recent_utterances: legacy.recent_utterances,
        }
    }
}

impl BrainState {
    pub fn new(config: BrainConfig) -> Result<Self, BrainError> {
        config.validate()?;
        Ok(Self {
            state_version: STATE_VERSION,
            config,
            next_node_id: 1,
            interaction_count: 0,
            learning_step_count: 0,
            training_example_count: 0,
            tokenizer: AdaptiveTokenizer::default(),
            nodes: BTreeMap::new(),
            edges: BTreeMap::new(),
            context_patterns: BTreeMap::new(),
            prompt_response_memory: BTreeMap::new(),
            recent_utterances: VecDeque::new(),
        })
    }
}

pub fn temporary_state_path(path: &std::path::Path) -> PathBuf {
    match path.file_name() {
        Some(file_name) => {
            let mut temp_name = file_name.to_os_string();
            temp_name.push(".tmp");
            path.with_file_name(temp_name)
        }
        None => path.with_extension("tmp"),
    }
}
