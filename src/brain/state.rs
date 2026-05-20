use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::PathBuf;

use super::cognition::{
    EpisodicMemory, NeuromodulatorState, ProceduralPattern, ProcedureSchema, SensoryFrame,
    WorkingMemoryState,
};
use super::config::BrainConfig;
use super::error::BrainError;
use super::tokenizer::AdaptiveTokenizer;

pub const STATE_VERSION: u32 = 4;
pub const LEGACY_STATE_VERSION_V3: u32 = 3;
pub const LEGACY_STATE_VERSION_V2: u32 = 2;
pub const LEGACY_STATE_VERSION_V1: u32 = 1;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum NodeKind {
    Sensor,
    Lexical,
    Phrase,
    Context,
    Concept,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum EdgeKind {
    Transition,
    ContextInput,
    ContextPrediction,
    ConceptMember,
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
    #[serde(default)]
    pub masked: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BrainEdge {
    pub source: u64,
    pub target: u64,
    pub kind: EdgeKind,
    pub strength: f32,
    pub activation_count: u64,
    pub last_activated_at: u64,
    #[serde(default)]
    pub masked: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContextPattern {
    pub token_ids: Vec<u64>,
    pub occurrence_count: u64,
    pub predicted_counts: BTreeMap<u64, u64>,
    pub node_id: Option<u64>,
    pub last_activated_at: u64,
    pub label: String,
    #[serde(default)]
    pub masked: bool,
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
    #[serde(default)]
    pub sensory_memory: VecDeque<SensoryFrame>,
    #[serde(default)]
    pub working_memory: WorkingMemoryState,
    #[serde(default)]
    pub episodic_memory: VecDeque<EpisodicMemory>,
    #[serde(default)]
    pub procedural_memory: BTreeMap<String, ProceduralPattern>,
    #[serde(default)]
    pub procedure_schemas: BTreeMap<String, ProcedureSchema>,
    #[serde(default)]
    pub neuromodulator: NeuromodulatorState,
    #[serde(skip, default)]
    pub prompt_inverted_index: BTreeMap<u64, BTreeSet<String>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LegacyBrainStateV3 {
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
            sensory_memory: VecDeque::new(),
            working_memory: WorkingMemoryState::default(),
            episodic_memory: VecDeque::new(),
            procedural_memory: BTreeMap::new(),
            procedure_schemas: BTreeMap::new(),
            neuromodulator: NeuromodulatorState::default(),
            prompt_inverted_index: BTreeMap::new(),
        }
    }
}

impl From<LegacyBrainStateV3> for BrainState {
    fn from(legacy: LegacyBrainStateV3) -> Self {
        Self {
            state_version: STATE_VERSION,
            config: legacy.config,
            next_node_id: legacy.next_node_id,
            interaction_count: legacy.interaction_count,
            learning_step_count: legacy.learning_step_count,
            training_example_count: legacy.training_example_count,
            tokenizer: legacy.tokenizer,
            nodes: legacy.nodes,
            edges: legacy.edges,
            context_patterns: legacy.context_patterns,
            prompt_response_memory: legacy.prompt_response_memory,
            recent_utterances: legacy.recent_utterances,
            sensory_memory: VecDeque::new(),
            working_memory: WorkingMemoryState::default(),
            episodic_memory: VecDeque::new(),
            procedural_memory: BTreeMap::new(),
            procedure_schemas: BTreeMap::new(),
            neuromodulator: NeuromodulatorState::default(),
            prompt_inverted_index: BTreeMap::new(),
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
            sensory_memory: VecDeque::new(),
            working_memory: WorkingMemoryState::default(),
            episodic_memory: VecDeque::new(),
            procedural_memory: BTreeMap::new(),
            procedure_schemas: BTreeMap::new(),
            neuromodulator: NeuromodulatorState::default(),
            prompt_inverted_index: BTreeMap::new(),
        }
    }
}

impl BrainState {
    pub fn new(config: BrainConfig) -> Result<Self, BrainError> {
        config.validate()?;
        let mut state = Self {
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
            sensory_memory: VecDeque::new(),
            working_memory: WorkingMemoryState::default(),
            episodic_memory: VecDeque::new(),
            procedural_memory: BTreeMap::new(),
            procedure_schemas: BTreeMap::new(),
            neuromodulator: NeuromodulatorState::default(),
            prompt_inverted_index: BTreeMap::new(),
        };
        state.register_tokenizer_nodes()?;
        Ok(state)
    }

    pub fn register_tokenizer_nodes(&mut self) -> Result<(), BrainError> {
        let mut max_id = 0;
        let entries = self.tokenizer.entries.clone();
        for (node_id, entry) in entries {
            if node_id > max_id {
                max_id = node_id;
            }
            if let std::collections::btree_map::Entry::Vacant(slot) = self.nodes.entry(node_id) {
                let kind = match entry.level {
                    super::tokenizer::TokenLevel::Sensor => NodeKind::Sensor,
                    super::tokenizer::TokenLevel::Word => NodeKind::Lexical,
                    super::tokenizer::TokenLevel::Phrase => NodeKind::Phrase,
                };
                slot.insert(BrainNode {
                    id: node_id,
                    label: entry.text.clone(),
                    kind,
                    activation_count: 0,
                    last_activated_at: 0,
                    salience: 0.0,
                    composition: Vec::new(),
                    masked: false,
                });
            }
        }
        self.next_node_id = self.next_node_id.max(max_id + 1);
        Ok(())
    }

    pub fn rebuild_inverted_index(&mut self) {
        let mut index: BTreeMap<u64, BTreeSet<String>> = BTreeMap::new();
        for prompt in self.prompt_response_memory.keys() {
            let tokens = self.tokenizer.tokenize(prompt);
            for token_id in tokens {
                index.entry(token_id).or_default().insert(prompt.clone());
            }
        }
        self.prompt_inverted_index = index;
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
