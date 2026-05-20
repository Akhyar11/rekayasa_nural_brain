use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum ResponseActionSource {
    ProceduralReasoning,
    ExactRecall,
    SimilarPrompt,
    EpisodicMemory,
    GraphContinuation,
    RecentMemory,
    ReflectiveFallback,
}

#[derive(Clone, Debug)]
pub struct ResponseActionCandidate {
    pub response: String,
    pub token_ids: Vec<u64>,
    pub source: ResponseActionSource,
    pub procedure_name: Option<String>,
    pub context_match: f32,
    pub episodic_match: f32,
    pub semantic_match: f32,
    pub novelty: f32,
    pub confidence: f32,
    pub historical_reward: f32,
    pub score: f32,
}

#[derive(Clone, Debug)]
pub struct ActionSelectionReport {
    pub chosen: ResponseActionCandidate,
    pub candidates: Vec<ResponseActionCandidate>,
    pub prediction_error: f32,
    pub predicted_outcomes: Vec<String>,
    pub predicted_procedures: Vec<String>,
    pub replayed_episodes: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SensoryFrame {
    pub interaction_index: u64,
    pub text: String,
    pub token_ids: Vec<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct WorkingMemoryState {
    pub active_text: String,
    pub active_token_ids: Vec<u64>,
    pub active_concept_ids: Vec<u64>,
    pub active_concepts: Vec<String>,
    pub predicted_intents: Vec<String>,
    pub predicted_outcomes: Vec<String>,
    pub predicted_procedures: Vec<String>,
    pub unresolved_goals: Vec<String>,
    pub resolved_goals: Vec<String>,
    pub emotional_tone: String,
    pub prediction_error: f32,
    pub confidence: f32,
    pub updated_at: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EpisodicMemory {
    pub interaction_index: u64,
    pub prompt: String,
    pub prompt_token_ids: Vec<u64>,
    pub response: String,
    pub response_token_ids: Vec<u64>,
    pub source: ResponseActionSource,
    pub reward: f32,
    pub novelty: f32,
    pub confidence: f32,
    pub semantic_match: f32,
    pub context_match: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct ProceduralPattern {
    pub use_count: u64,
    pub total_reward: f32,
    pub last_used_at: u64,
}

impl ProceduralPattern {
    pub fn average_reward(&self) -> f32 {
        if self.use_count == 0 {
            0.0
        } else {
            self.total_reward / self.use_count as f32
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum ProcedureKind {
    ArithmeticAddition,
    ArithmeticSubtraction,
    ArithmeticMultiplication,
    ArithmeticDivision,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProcedureSchema {
    pub name: String,
    pub kind: ProcedureKind,
    pub evidence_count: u64,
    pub use_count: u64,
    pub success_count: u64,
    pub total_reward: f32,
    pub last_used_at: u64,
    pub last_prediction_error: f32,
}

impl ProcedureSchema {
    pub fn confidence(&self) -> f32 {
        let evidence = (self.evidence_count.min(8) as f32) / 8.0;
        let success = if self.use_count == 0 {
            0.5
        } else {
            self.success_count as f32 / self.use_count as f32
        };
        (evidence * 0.6 + success * 0.4).clamp(0.0, 1.0)
    }

    pub fn average_reward(&self) -> f32 {
        if self.use_count == 0 {
            0.0
        } else {
            self.total_reward / self.use_count as f32
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NeuromodulatorState {
    pub dopamine: f32,
    pub acetylcholine: f32,
    pub serotonin: f32,
}

impl Default for NeuromodulatorState {
    fn default() -> Self {
        Self {
            dopamine: 0.5,
            acetylcholine: 0.5,
            serotonin: 0.5,
        }
    }
}

impl NeuromodulatorState {
    pub fn apply_feedback(
        &mut self,
        reward: f32,
        novelty: f32,
        confidence: f32,
        prediction_error: f32,
    ) {
        let reward = reward.clamp(0.0, 1.0);
        let novelty = novelty.clamp(0.0, 1.0);
        let prediction_error = prediction_error.clamp(0.0, 1.0);
        let stability = ((reward + confidence.clamp(0.0, 1.0)) * 0.5 * (1.0 - prediction_error))
            .clamp(0.0, 1.0);
        self.dopamine = blend(self.dopamine, reward * (1.0 - prediction_error * 0.5), 0.35);
        self.acetylcholine = blend(self.acetylcholine, novelty.max(prediction_error), 0.35);
        self.serotonin = blend(self.serotonin, stability, 0.25);
    }
}

fn blend(current: f32, target: f32, rate: f32) -> f32 {
    (current + (target - current) * rate).clamp(0.0, 1.0)
}
