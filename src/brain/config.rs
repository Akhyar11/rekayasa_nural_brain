use serde::{Deserialize, Serialize};
use super::error::BrainError;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct GenerationConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub min_confidence: f32,
    pub context_weight: f32,
    pub transition_weight: f32,
    pub recent_memory_weight: f32,
    pub soft_recall_weight: f32,
    pub randomness_seed: Option<u64>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 20,
            top_p: 0.9,
            repetition_penalty: 1.2,
            min_confidence: 0.0001,
            context_weight: 1.0,
            transition_weight: 0.6,
            recent_memory_weight: 0.3,
            soft_recall_weight: 1.2,
            randomness_seed: None,
        }
    }
}

impl GenerationConfig {
    pub fn validate(&self) -> Result<(), BrainError> {
        if self.temperature < 0.0 || !self.temperature.is_finite() {
            return Err(BrainError::InvalidConfig("temperature harus >= 0.0"));
        }
        if self.top_k == 0 {
            return Err(BrainError::InvalidConfig("top_k harus >= 1"));
        }
        if !(0.0..=1.0).contains(&self.top_p) || !self.top_p.is_finite() {
            return Err(BrainError::InvalidConfig("top_p harus berada di antara 0.0 dan 1.0"));
        }
        if self.repetition_penalty < 1.0 || !self.repetition_penalty.is_finite() {
            return Err(BrainError::InvalidConfig("repetition_penalty harus >= 1.0"));
        }
        if self.min_confidence < 0.0 || !self.min_confidence.is_finite() {
            return Err(BrainError::InvalidConfig("min_confidence harus >= 0.0"));
        }
        Ok(())
    }
}

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
    #[serde(default)]
    pub generation_config: GenerationConfig,
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
            generation_config: GenerationConfig::default(),
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
        self.generation_config.validate()?;

        Ok(())
    }
}
