use super::state::{BrainEdge, BrainState, EdgeKind, NodeKind};
use super::tokenizer::TokenLevel;

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
    pub learning_steps: u64,
    pub training_examples: u64,
    pub token_count: usize,
    pub sensor_token_count: usize,
    pub word_token_count: usize,
    pub phrase_token_count: usize,
    pub context_node_count: usize,
    pub edge_count: usize,
    pub remembered_utterances: usize,
    pub sensory_frames: usize,
    pub episodic_memories: usize,
    pub procedural_patterns: usize,
    pub procedure_schemas: usize,
    pub working_memory_tokens: usize,
    pub working_memory_concepts: usize,
    pub unresolved_goals: usize,
    pub predicted_procedures: usize,
    pub prediction_error: f32,
    pub emotional_tone: String,
    pub dopamine: f32,
    pub acetylcholine: f32,
    pub serotonin: f32,
    pub latest_tokens: Vec<String>,
    pub strongest_edges: Vec<BrainEdgeSummary>,
}

impl BrainState {
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

        let mut latest_tokens: Vec<&super::tokenizer::TokenEntry> =
            self.tokenizer.entries.values().collect();
        latest_tokens.sort_by(|left, right| {
            right
                .created_at
                .cmp(&left.created_at)
                .then_with(|| right.node_id.cmp(&left.node_id))
        });

        BrainSummary {
            interactions: self.interaction_count,
            learning_steps: self.learning_step_count,
            training_examples: self.training_example_count,
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
            sensory_frames: self.sensory_memory.len(),
            episodic_memories: self.episodic_memory.len(),
            procedural_patterns: self.procedural_memory.len(),
            procedure_schemas: self.procedure_schemas.len(),
            working_memory_tokens: self.working_memory.active_token_ids.len(),
            working_memory_concepts: self.working_memory.active_concept_ids.len(),
            unresolved_goals: self.working_memory.unresolved_goals.len(),
            predicted_procedures: self.working_memory.predicted_procedures.len(),
            prediction_error: self.working_memory.prediction_error,
            emotional_tone: self.working_memory.emotional_tone.clone(),
            dopamine: self.neuromodulator.dopamine,
            acetylcholine: self.neuromodulator.acetylcholine,
            serotonin: self.neuromodulator.serotonin,
            latest_tokens: latest_tokens
                .into_iter()
                .take(8)
                .map(|token| token.text.clone())
                .collect(),
            strongest_edges,
        }
    }
}
