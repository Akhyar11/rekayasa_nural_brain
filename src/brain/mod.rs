pub mod config;
pub mod error;
pub mod generation;
pub mod learning;
pub mod persistence;
pub mod probability;
pub mod pruning;
pub mod retrieval;
pub mod sampling;
pub mod similarity;
pub mod state;
pub mod summary;
pub mod tokenizer;

#[cfg(test)]
mod tests;

pub use config::{BrainConfig, GenerationConfig};
pub use error::BrainError;
pub use generation::{CandidateSource, InteractionReport, TokenCandidate};
pub use learning::{LearningReport, TrainingExampleReport};
pub use probability::{calculate_smoothed_probability, entropy};
pub use retrieval::SimilarPrompt;
pub use sampling::{sample_next_token, SimpleRng};
pub use similarity::{
    jaccard_similarity, token_overlap_similarity, RandomProjectionVector,
};
pub use state::{BrainEdge, BrainNode, BrainState, EdgeKind, NodeKind, STATE_VERSION};
pub use summary::{BrainEdgeSummary, BrainSummary};
pub use tokenizer::{AdaptiveTokenizer, TokenEntry, TokenLevel};
