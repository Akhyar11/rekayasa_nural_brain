use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

use super::cognition::{
    ActionSelectionReport, EpisodicMemory, ResponseActionCandidate, ResponseActionSource,
    SensoryFrame,
};
use super::config::GenerationConfig;
use super::error::BrainError;
use super::learning::{
    ArithmeticQuery, LearningReport, compute_arithmetic, context_key, normalize_input,
    parse_arithmetic_query, procedure_name,
};
use super::sampling::SimpleRng;
use super::similarity::{jaccard_similarity, token_overlap_similarity};
use super::state::{BrainEdge, BrainState, EdgeKind, NodeKind, UtteranceMemory};
use super::tokenizer::{TokenLevel, collapse_whitespace};

#[derive(Clone, Debug)]
pub struct InteractionReport {
    pub learning: LearningReport,
    pub response: String,
    pub response_source: ResponseActionSource,
    pub selected_procedure: Option<String>,
    pub response_score: f32,
    pub prediction_error: f32,
    pub replayed_episodes: usize,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum CandidateSource {
    ContextPattern,
    TransitionEdge,
    RecentMemory,
    SoftRecall,
    ExactRecall,
    Mixed,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TokenCandidate {
    pub token_id: u64,
    pub score: f32,
    pub probability: f32,
    pub occurrences: u64,
    pub source: CandidateSource,
}

#[derive(Default)]
struct ComponentScores {
    context_score: f32,
    context_weight_sum: f32,
    transition_score: f32,
    recent_memory_score: f32,
    soft_recall_score: f32,
    sources: BTreeSet<CandidateSource>,
}

#[derive(Clone, Debug)]
struct GeneratedSequence {
    text: String,
    token_ids: Vec<u64>,
    confidence: f32,
}

impl BrainState {
    pub fn interact_with_seed(
        &mut self,
        input: &str,
        seed: Option<u64>,
        gen_config: &GenerationConfig,
    ) -> Result<InteractionReport, BrainError> {
        let learning = self.learn_text(input)?;
        self.interaction_count += 1; // Live interaction
        self.record_sensory_frame(
            self.interaction_count,
            &learning.normalized_text,
            &learning.token_ids,
        );

        let mut selection = self.generate_action_selection_from_tokens_with_seed(
            &learning.normalized_text,
            &learning.token_ids,
            seed,
            gen_config,
        )?;
        let replayed =
            self.apply_action_outcome(&learning.normalized_text, &learning.token_ids, &selection)?;
        selection.replayed_episodes = replayed;
        self.update_working_memory(&learning.normalized_text, &learning.token_ids, &selection);

        Ok(InteractionReport {
            learning,
            response: selection.chosen.response.clone(),
            response_source: selection.chosen.source,
            selected_procedure: selection.chosen.procedure_name.clone(),
            response_score: selection.chosen.score,
            prediction_error: selection.prediction_error,
            replayed_episodes: selection.replayed_episodes,
        })
    }

    pub fn interact(&mut self, input: &str) -> Result<InteractionReport, BrainError> {
        let gen_config = self.config.generation_config;
        self.interact_with_seed(input, None, &gen_config)
    }

    pub fn generate_response_with_seed(
        &self,
        prompt: &str,
        seed: Option<u64>,
        gen_config: &GenerationConfig,
    ) -> Result<String, BrainError> {
        let normalized_text = normalize_input(prompt)?;
        let token_ids = self.tokenizer.tokenize(&normalized_text);
        let selection = self.generate_action_selection_from_tokens_with_seed(
            &normalized_text,
            &token_ids,
            seed,
            gen_config,
        )?;
        Ok(selection.chosen.response)
    }

    pub fn generate_response(&self, prompt: &str) -> Result<String, BrainError> {
        self.generate_response_with_seed(prompt, None, &self.config.generation_config)
    }

    pub fn generate_response_from_tokens(
        &self,
        prompt_token_ids: &[u64],
    ) -> Result<String, BrainError> {
        self.generate_response_from_tokens_with_seed(
            prompt_token_ids,
            None,
            &self.config.generation_config,
        )
    }

    pub fn generate_response_from_tokens_with_seed(
        &self,
        prompt_token_ids: &[u64],
        seed: Option<u64>,
        gen_config: &GenerationConfig,
    ) -> Result<String, BrainError> {
        let prompt_text = self.describe_tokens(prompt_token_ids);
        let selection = self.generate_action_selection_from_tokens_with_seed(
            &prompt_text,
            prompt_token_ids,
            seed,
            gen_config,
        )?;
        Ok(selection.chosen.response)
    }

    pub fn generate_action_selection_with_seed(
        &self,
        prompt: &str,
        seed: Option<u64>,
        gen_config: &GenerationConfig,
    ) -> Result<ActionSelectionReport, BrainError> {
        let normalized_text = normalize_input(prompt)?;
        let token_ids = self.tokenizer.tokenize(&normalized_text);
        self.generate_action_selection_from_tokens_with_seed(
            &normalized_text,
            &token_ids,
            seed,
            gen_config,
        )
    }

    pub fn generate_action_selection(
        &self,
        prompt: &str,
    ) -> Result<ActionSelectionReport, BrainError> {
        self.generate_action_selection_with_seed(prompt, None, &self.config.generation_config)
    }

    pub fn generate_action_selection_from_tokens_with_seed(
        &self,
        prompt_text: &str,
        prompt_token_ids: &[u64],
        seed: Option<u64>,
        gen_config: &GenerationConfig,
    ) -> Result<ActionSelectionReport, BrainError> {
        if prompt_token_ids.is_empty() {
            let fallback = self.reflective_fallback(prompt_text, prompt_token_ids);
            return Ok(ActionSelectionReport {
                chosen: fallback.clone(),
                candidates: vec![fallback],
                prediction_error: 1.0,
                predicted_outcomes: Vec::new(),
                predicted_procedures: Vec::new(),
                replayed_episodes: 0,
            });
        }

        let mut candidates = self.collect_response_action_candidates(
            prompt_text,
            prompt_token_ids,
            seed,
            gen_config,
        );
        if candidates.is_empty() {
            candidates.push(self.reflective_fallback(prompt_text, prompt_token_ids));
        }

        let procedural_query = parse_arithmetic_query(prompt_text);
        for candidate in &mut candidates {
            candidate.semantic_match =
                self.semantic_alignment(prompt_token_ids, &candidate.token_ids);
            candidate.novelty = self.response_novelty(&candidate.token_ids);
            candidate.historical_reward = self
                .procedural_memory
                .get(&candidate.response)
                .map(|pattern| pattern.average_reward())
                .unwrap_or(0.0);

            let novelty_drive = candidate.novelty * (0.5 + self.neuromodulator.acetylcholine * 0.5);
            let reward_drive =
                candidate.historical_reward * (0.5 + self.neuromodulator.dopamine * 0.5);
            let confidence_drive =
                candidate.confidence * (0.5 + self.neuromodulator.serotonin * 0.5);
            let procedural_bonus = if procedural_query.is_some()
                && candidate.source == ResponseActionSource::ProceduralReasoning
            {
                0.4 + candidate.confidence * 0.3
            } else {
                0.0
            };

            candidate.score = candidate.context_match * gen_config.action_context_weight
                + candidate.episodic_match * gen_config.action_episodic_weight
                + candidate.semantic_match * gen_config.action_semantic_weight
                + novelty_drive * gen_config.action_novelty_weight
                + confidence_drive * gen_config.action_confidence_weight
                + reward_drive * gen_config.action_reward_weight
                + procedural_bonus;
        }

        candidates.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| right.confidence.total_cmp(&left.confidence))
                .then_with(|| left.response.len().cmp(&right.response.len()))
        });

        let mut seen_responses = BTreeSet::new();
        candidates.retain(|candidate| seen_responses.insert(candidate.response.clone()));
        if candidates.len() > gen_config.action_candidate_limit {
            candidates.truncate(gen_config.action_candidate_limit);
        }

        if candidates.is_empty() {
            let fallback = self.reflective_fallback(prompt_text, prompt_token_ids);
            return Ok(ActionSelectionReport {
                chosen: fallback.clone(),
                candidates: vec![fallback],
                prediction_error: 1.0,
                predicted_outcomes: Vec::new(),
                predicted_procedures: Vec::new(),
                replayed_episodes: 0,
            });
        }

        let predicted_outcomes = candidates
            .iter()
            .take(3)
            .map(|candidate| candidate.response.clone())
            .collect();
        let predicted_procedures = unique_procedures(&candidates);
        let chosen = candidates[0].clone();
        let prediction_error = estimate_prediction_error(
            prompt_text,
            prompt_token_ids,
            &candidates,
            &chosen,
            &predicted_procedures,
        );
        Ok(ActionSelectionReport {
            chosen,
            candidates,
            prediction_error,
            predicted_outcomes,
            predicted_procedures,
            replayed_episodes: 0,
        })
    }

    fn collect_response_action_candidates(
        &self,
        prompt_text: &str,
        prompt_token_ids: &[u64],
        seed: Option<u64>,
        gen_config: &GenerationConfig,
    ) -> Vec<ResponseActionCandidate> {
        let mut candidates = Vec::new();

        if let Some(query) = parse_arithmetic_query(prompt_text)
            && let Some(candidate) = self.build_procedural_candidate(prompt_text, query)
        {
            candidates.push(candidate);
        }

        if let Some(responses) = self.prompt_response_memory.get(prompt_text) {
            let total: u64 = responses.values().sum();
            let mut ranked: Vec<(&String, &u64)> = responses.iter().collect();
            ranked.sort_by(
                |(left_response, left_count), (right_response, right_count)| {
                    right_count
                        .cmp(left_count)
                        .then_with(|| left_response.len().cmp(&right_response.len()))
                },
            );
            for (response, count) in ranked.into_iter().take(2) {
                candidates.push(ResponseActionCandidate {
                    response: response.clone(),
                    token_ids: self.tokenizer.tokenize(response),
                    source: ResponseActionSource::ExactRecall,
                    procedure_name: None,
                    context_match: 1.0,
                    episodic_match: 0.55,
                    semantic_match: 0.0,
                    novelty: 0.0,
                    confidence: if total == 0 {
                        0.0
                    } else {
                        *count as f32 / total as f32
                    },
                    historical_reward: 0.0,
                    score: 0.0,
                });
            }
        }

        let mut similar_prompts = self.find_similar_prompts(prompt_token_ids);
        similar_prompts.sort_by(|left, right| right.similarity.total_cmp(&left.similarity));
        for similar in similar_prompts.into_iter().take(2) {
            candidates.push(ResponseActionCandidate {
                response: similar.response.clone(),
                token_ids: self.tokenizer.tokenize(&similar.response),
                source: ResponseActionSource::SimilarPrompt,
                procedure_name: None,
                context_match: similar.similarity,
                episodic_match: 0.35 * similar.similarity,
                semantic_match: 0.0,
                novelty: 0.0,
                confidence: similar.similarity,
                historical_reward: 0.0,
                score: 0.0,
            });
        }

        let mut episodes: Vec<(f32, &EpisodicMemory)> = self
            .episodic_memory
            .iter()
            .filter_map(|episode| {
                let similarity = jaccard_similarity(prompt_token_ids, &episode.prompt_token_ids);
                if similarity >= 0.2 {
                    Some((similarity, episode))
                } else {
                    None
                }
            })
            .collect();
        episodes.sort_by(|left, right| {
            right
                .0
                .total_cmp(&left.0)
                .then_with(|| right.1.reward.total_cmp(&left.1.reward))
        });
        for (similarity, episode) in episodes.into_iter().take(2) {
            candidates.push(ResponseActionCandidate {
                response: episode.response.clone(),
                token_ids: episode.response_token_ids.clone(),
                source: ResponseActionSource::EpisodicMemory,
                procedure_name: None,
                context_match: similarity,
                episodic_match: (similarity + episode.reward + episode.confidence) / 3.0,
                semantic_match: 0.0,
                novelty: 0.0,
                confidence: (episode.confidence + episode.reward) * 0.5,
                historical_reward: 0.0,
                score: 0.0,
            });
        }

        for episode in self.episodic_memory.iter().rev().take(2) {
            candidates.push(ResponseActionCandidate {
                response: episode.response.clone(),
                token_ids: episode.response_token_ids.clone(),
                source: ResponseActionSource::RecentMemory,
                procedure_name: None,
                context_match: jaccard_similarity(prompt_token_ids, &episode.prompt_token_ids),
                episodic_match: episode.reward * 0.5,
                semantic_match: 0.0,
                novelty: 0.0,
                confidence: episode.confidence * 0.8,
                historical_reward: 0.0,
                score: 0.0,
            });
        }

        for offset in 0..gen_config.graph_candidate_count {
            let candidate_seed = seed_with_offset(seed, offset);
            if let Ok(Some(sequence)) = self.generate_graph_sequence_from_tokens_with_seed(
                prompt_token_ids,
                candidate_seed,
                gen_config,
            ) {
                candidates.push(ResponseActionCandidate {
                    response: sequence.text,
                    token_ids: sequence.token_ids,
                    source: ResponseActionSource::GraphContinuation,
                    procedure_name: None,
                    context_match: (0.35 + sequence.confidence * 0.65).clamp(0.0, 1.0),
                    episodic_match: 0.15,
                    semantic_match: 0.0,
                    novelty: 0.0,
                    confidence: sequence.confidence,
                    historical_reward: 0.0,
                    score: 0.0,
                });
            }
        }

        candidates
    }

    fn build_procedural_candidate(
        &self,
        prompt_text: &str,
        query: ArithmeticQuery,
    ) -> Option<ResponseActionCandidate> {
        let procedure = self.procedure_schemas.get(procedure_name(query.kind))?;
        let result = compute_arithmetic(query.kind, query.left, query.right)?;
        let response = format_procedural_response(prompt_text, result);
        let token_ids = self.tokenizer.tokenize(&response);
        let confidence = procedure.confidence();
        let episodic_match = (confidence * 0.7 + procedure.average_reward() * 0.3).clamp(0.0, 1.0);

        Some(ResponseActionCandidate {
            response,
            token_ids,
            source: ResponseActionSource::ProceduralReasoning,
            procedure_name: Some(procedure.name.clone()),
            context_match: 1.0,
            episodic_match,
            semantic_match: 0.0,
            novelty: 0.0,
            confidence,
            historical_reward: procedure.average_reward(),
            score: 0.0,
        })
    }

    fn generate_graph_sequence_from_tokens_with_seed(
        &self,
        prompt_token_ids: &[u64],
        seed: Option<u64>,
        gen_config: &GenerationConfig,
    ) -> Result<Option<GeneratedSequence>, BrainError> {
        let mut generated = Vec::new();
        let mut context = self.expand_context_tokens(prompt_token_ids);
        if context.is_empty() {
            context = prompt_token_ids.to_vec();
        }
        let mut step_confidences = Vec::new();
        let mut seen = BTreeSet::new();
        let mut rng = SimpleRng::new(seed);

        for _ in 0..self.config.response_token_limit {
            let mut candidates =
                self.next_token_distribution(&context, prompt_token_ids, &generated, gen_config);
            if generated.len() >= 2 {
                let penultimate = generated[generated.len() - 2];
                let last = generated[generated.len() - 1];
                let mut blocked_tokens = BTreeSet::new();
                for window in generated.windows(3) {
                    if window[0] == penultimate && window[1] == last {
                        blocked_tokens.insert(window[2]);
                    }
                }
                if !blocked_tokens.is_empty() {
                    let filtered: Vec<TokenCandidate> = candidates
                        .iter()
                        .filter(|candidate| !blocked_tokens.contains(&candidate.token_id))
                        .cloned()
                        .collect();
                    if !filtered.is_empty() {
                        candidates = filtered;
                        let total_p: f32 = candidates
                            .iter()
                            .map(|candidate| candidate.probability)
                            .sum();
                        if total_p > 0.0 {
                            for candidate in &mut candidates {
                                candidate.probability /= total_p;
                            }
                        }
                    }
                }
            }

            if candidates.is_empty() || candidates[0].probability < gen_config.min_confidence {
                break;
            }

            let next_token = match self.sample_token(&candidates, &mut rng, gen_config) {
                Some(token_id) => token_id,
                None => break,
            };

            let selected_probability = candidates
                .iter()
                .find(|candidate| candidate.token_id == next_token)
                .map(|candidate| candidate.probability)
                .unwrap_or(0.0);
            step_confidences.push(selected_probability);

            let max_win = self.config.max_context_window;
            let current_window = &context[context.len().saturating_sub(max_win)..];
            if !seen.insert((context_key(current_window), next_token)) {
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
            return Ok(None);
        }

        let decoded = self.tokenizer.decode_tokens(&generated)?;
        if decoded.is_empty() {
            return Ok(None);
        }

        let confidence = if step_confidences.is_empty() {
            0.0
        } else {
            step_confidences.iter().sum::<f32>() / step_confidences.len() as f32
        };

        Ok(Some(GeneratedSequence {
            text: decoded,
            token_ids: generated,
            confidence,
        }))
    }

    fn reflective_fallback(
        &self,
        prompt_text: &str,
        prompt_token_ids: &[u64],
    ) -> ResponseActionCandidate {
        let focus = if !self.working_memory.active_concepts.is_empty() {
            self.working_memory.active_concepts.join(", ")
        } else if !prompt_text.is_empty() {
            prompt_text.to_string()
        } else {
            self.describe_tokens(prompt_token_ids)
        };
        let response = if focus.is_empty() {
            "saya belum punya aksi bahasa yang stabil untuk konteks ini.".to_string()
        } else {
            format!("saya memahami konteks {focus}, tetapi belum punya aksi bahasa yang stabil.")
        };

        ResponseActionCandidate {
            token_ids: self.tokenizer.tokenize(&response),
            response,
            source: ResponseActionSource::ReflectiveFallback,
            procedure_name: None,
            context_match: 0.35,
            episodic_match: 0.1,
            semantic_match: 0.0,
            novelty: 0.5,
            confidence: 0.2,
            historical_reward: 0.0,
            score: 0.0,
        }
    }

    fn semantic_alignment(&self, prompt_token_ids: &[u64], response_token_ids: &[u64]) -> f32 {
        let prompt_concepts = self.collect_concepts(prompt_token_ids);
        let response_concepts = self.collect_concepts(response_token_ids);
        if !prompt_concepts.is_empty() && !response_concepts.is_empty() {
            let intersection = prompt_concepts.intersection(&response_concepts).count();
            let union = prompt_concepts.union(&response_concepts).count();
            if union > 0 {
                return intersection as f32 / union as f32;
            }
        }
        token_overlap_similarity(prompt_token_ids, response_token_ids).clamp(0.0, 1.0)
    }

    fn response_novelty(&self, response_token_ids: &[u64]) -> f32 {
        if self.episodic_memory.is_empty() {
            return 0.5;
        }

        let mut max_similarity = 0.0f32;
        for episode in self.episodic_memory.iter().rev().take(24) {
            let similarity = jaccard_similarity(response_token_ids, &episode.response_token_ids);
            if similarity > max_similarity {
                max_similarity = similarity;
            }
        }

        (1.0 - max_similarity).clamp(0.0, 1.0)
    }

    fn collect_concepts(&self, token_ids: &[u64]) -> BTreeSet<u64> {
        let mut concepts = BTreeSet::new();
        for edge in self.edges.values() {
            if edge.masked || edge.kind != EdgeKind::ConceptMember {
                continue;
            }
            if token_ids.contains(&edge.source) {
                concepts.insert(edge.target);
            }
        }
        concepts
    }

    fn record_sensory_frame(&mut self, interaction_index: u64, text: &str, token_ids: &[u64]) {
        if self.sensory_memory.len() >= self.config.sensory_memory_capacity {
            self.sensory_memory.pop_front();
        }
        self.sensory_memory.push_back(SensoryFrame {
            interaction_index,
            text: text.to_string(),
            token_ids: token_ids.to_vec(),
        });
    }

    fn update_working_memory(
        &mut self,
        prompt_text: &str,
        prompt_token_ids: &[u64],
        selection: &ActionSelectionReport,
    ) {
        let mut active_token_ids = self.expand_context_tokens(prompt_token_ids);
        if active_token_ids.is_empty() {
            active_token_ids = prompt_token_ids.to_vec();
        }
        if active_token_ids.len() > self.config.working_memory_capacity {
            active_token_ids = active_token_ids
                [active_token_ids.len() - self.config.working_memory_capacity..]
                .to_vec();
        }

        let active_concept_ids: Vec<u64> = self
            .collect_concepts(&active_token_ids)
            .into_iter()
            .collect();
        let active_concepts = active_concept_ids
            .iter()
            .map(|concept_id| self.node_label(*concept_id))
            .collect();

        self.working_memory.active_text = prompt_text.to_string();
        self.working_memory.active_token_ids = active_token_ids;
        self.working_memory.active_concept_ids = active_concept_ids;
        self.working_memory.active_concepts = active_concepts;
        self.working_memory.predicted_intents = infer_intents(prompt_text);
        self.working_memory.predicted_outcomes = selection.predicted_outcomes.clone();
        self.working_memory.predicted_procedures = selection.predicted_procedures.clone();
        let goals = infer_goals(prompt_text, &self.working_memory.predicted_intents);
        if selection.chosen.source == ResponseActionSource::ReflectiveFallback
            || (selection.prediction_error > 0.45
                && selection.chosen.source != ResponseActionSource::ProceduralReasoning)
        {
            self.working_memory.unresolved_goals = goals;
            self.working_memory.resolved_goals.clear();
        } else {
            self.working_memory.resolved_goals = goals;
            self.working_memory.unresolved_goals.clear();
        }
        self.working_memory.emotional_tone = infer_emotional_tone(prompt_text);
        self.working_memory.prediction_error = selection.prediction_error;
        self.working_memory.confidence = selection.chosen.confidence.clamp(0.0, 1.0);
        self.working_memory.updated_at = self.interaction_count;
    }

    fn apply_action_outcome(
        &mut self,
        prompt_text: &str,
        prompt_token_ids: &[u64],
        selection: &ActionSelectionReport,
    ) -> Result<usize, BrainError> {
        let step = self.learning_step_count.max(1);
        let chosen = &selection.chosen;
        let reward = intrinsic_reward(chosen, selection.prediction_error);

        for token_id in &chosen.token_ids {
            self.activate_node(*token_id, step)?;
            self.tokenizer.touch_token(*token_id, step)?;
        }

        if let (Some(&prompt_last), Some(&response_first)) =
            (prompt_token_ids.last(), chosen.token_ids.first())
        {
            self.strengthen_edge(
                prompt_last,
                response_first,
                EdgeKind::Transition,
                step,
                0.35 + reward * 0.65,
            );
        }
        self.learn_transitions(&chosen.token_ids, step);

        if reward >= self.config.self_reward_threshold
            && chosen.source != ResponseActionSource::GraphContinuation
        {
            self.store_prompt_response_pair(prompt_text, &chosen.response);
        }

        for rejected in selection.candidates.iter().skip(1).take(3) {
            if let (Some(&prompt_last), Some(&response_first)) =
                (prompt_token_ids.last(), rejected.token_ids.first())
            {
                let error_signal = (chosen.score - rejected.score).max(0.05);
                self.weaken_edge(
                    prompt_last,
                    response_first,
                    EdgeKind::Transition,
                    error_signal.min(1.0),
                );
            }
        }

        self.store_episode(prompt_text, prompt_token_ids, chosen, reward);
        self.update_procedural_memory(&chosen.response, reward, step);
        if let Some(procedure_name) = chosen.procedure_name.as_deref() {
            self.update_procedure_schema(procedure_name, reward, step, selection.prediction_error);
        }
        self.neuromodulator.apply_feedback(
            reward,
            chosen.novelty,
            chosen.confidence,
            selection.prediction_error,
        );

        let replayed = if self
            .interaction_count
            .is_multiple_of(self.config.replay_interval)
        {
            self.replay_episodes(step)?
        } else {
            0
        };

        Ok(replayed)
    }

    fn store_episode(
        &mut self,
        prompt_text: &str,
        prompt_token_ids: &[u64],
        chosen: &ResponseActionCandidate,
        reward: f32,
    ) {
        if self.episodic_memory.len() >= self.config.episodic_memory_capacity {
            self.episodic_memory.pop_front();
        }
        self.episodic_memory.push_back(EpisodicMemory {
            interaction_index: self.interaction_count,
            prompt: prompt_text.to_string(),
            prompt_token_ids: prompt_token_ids.to_vec(),
            response: chosen.response.clone(),
            response_token_ids: chosen.token_ids.clone(),
            source: chosen.source,
            reward,
            novelty: chosen.novelty,
            confidence: chosen.confidence,
            semantic_match: chosen.semantic_match,
            context_match: chosen.context_match,
        });
    }

    fn update_procedural_memory(&mut self, response: &str, reward: f32, step: u64) {
        let pattern = self
            .procedural_memory
            .entry(response.to_string())
            .or_default();
        pattern.use_count += 1;
        pattern.total_reward += reward;
        pattern.last_used_at = step;
    }

    fn update_procedure_schema(
        &mut self,
        procedure_name: &str,
        reward: f32,
        step: u64,
        prediction_error: f32,
    ) {
        let Some(schema) = self.procedure_schemas.get_mut(procedure_name) else {
            return;
        };
        schema.use_count += 1;
        if prediction_error <= 0.25 {
            schema.success_count += 1;
        }
        schema.total_reward += reward;
        schema.last_used_at = step;
        schema.last_prediction_error = prediction_error;
    }

    fn replay_episodes(&mut self, step: u64) -> Result<usize, BrainError> {
        let mut episodes: Vec<EpisodicMemory> = self.episodic_memory.iter().cloned().collect();
        episodes.sort_by(|left, right| {
            let left_priority = left.reward + left.confidence * 0.25 + left.semantic_match * 0.2;
            let right_priority =
                right.reward + right.confidence * 0.25 + right.semantic_match * 0.2;
            right_priority
                .total_cmp(&left_priority)
                .then_with(|| right.interaction_index.cmp(&left.interaction_index))
        });

        let mut replayed = 0;
        for episode in episodes.into_iter().take(self.config.replay_batch_size) {
            for token_id in &episode.response_token_ids {
                self.activate_node(*token_id, step)?;
                self.tokenizer.touch_token(*token_id, step)?;
            }
            if let (Some(&prompt_last), Some(&response_first)) = (
                episode.prompt_token_ids.last(),
                episode.response_token_ids.first(),
            ) {
                self.strengthen_edge(
                    prompt_last,
                    response_first,
                    EdgeKind::Transition,
                    step,
                    0.2 + episode.reward * 0.5,
                );
            }
            self.learn_transitions(&episode.response_token_ids, step);
            if episode.reward >= self.config.self_reward_threshold
                && episode.source != ResponseActionSource::GraphContinuation
            {
                self.store_prompt_response_pair(&episode.prompt, &episode.response);
            }
            replayed += 1;
        }

        Ok(replayed)
    }

    pub fn next_token_distribution(
        &self,
        context: &[u64],
        prompt: &[u64],
        generated: &[u64],
        gen_config: &GenerationConfig,
    ) -> Vec<TokenCandidate> {
        let prompt_tokens = prompt.to_vec();
        let g_len = generated.len();

        // 1. Soft recall via Inverted Index
        let mut candidate_prompts = BTreeSet::new();
        for &token_id in &prompt_tokens {
            if let Some(prompts) = self.prompt_inverted_index.get(&token_id) {
                candidate_prompts.extend(prompts);
            }
        }

        let soft_recall_results: Vec<(u64, f32)> = candidate_prompts
            .into_iter()
            .par_bridge()
            .filter_map(|key_prompt| {
                if let Some(responses) = self.prompt_response_memory.get(key_prompt.as_str()) {
                    let key_tokens = self.tokenizer.tokenize(key_prompt);
                    let sim = jaccard_similarity(&prompt_tokens, &key_tokens);
                    if sim >= 0.25
                        && let Some((response_text, _)) =
                            responses
                                .iter()
                                .max_by(|(left_r, left_c), (right_r, right_c)| {
                                    left_c
                                        .cmp(right_c)
                                        .then_with(|| right_r.len().cmp(&left_r.len()))
                                })
                    {
                        let response_tokens = self.tokenizer.tokenize(response_text);
                        if g_len < response_tokens.len() {
                            return Some((response_tokens[g_len], sim));
                        }
                    }
                }
                None
            })
            .collect();

        // 2. Recent memory in parallel
        let utterances: Vec<&UtteranceMemory> = self.recent_utterances.iter().collect();
        let recent_memory_results: Vec<(u64, f32)> = utterances
            .par_iter()
            .enumerate()
            .flat_map(|(idx, utterance)| {
                let mut local = Vec::new();
                let n = utterances.len();
                let recency_weight = (idx + 1) as f32 / n as f32;
                if let Some(&last_token) = context.last() {
                    for i in 0..utterance.token_ids.len() {
                        if utterance.token_ids[i] == last_token && i + 1 < utterance.token_ids.len()
                        {
                            local.push((utterance.token_ids[i + 1], recency_weight));
                        }
                    }
                }
                for &t in &utterance.token_ids {
                    local.push((t, 0.05 * recency_weight));
                }
                local
            })
            .collect();

        // 3. Context pattern in parallel
        let max_window = self.config.max_context_window.min(context.len());
        let windows: Vec<Vec<u64>> = (1..=max_window)
            .map(|w| context[context.len() - w..].to_vec())
            .collect();
        let context_results: Vec<(u64, f32, f32)> = windows
            .par_iter()
            .enumerate()
            .flat_map(|(index, prefix)| {
                let w = (index + 1) as f32;
                let mut local = Vec::new();
                if let Some(pattern) = self.context_patterns.get(&context_key(prefix))
                    && !pattern.masked
                {
                    let total_predictions: u64 = pattern.predicted_counts.values().sum();
                    if total_predictions > 0 {
                        for (&token_id, &count) in &pattern.predicted_counts {
                            if let Some(tok) = self.tokenizer.entries.get(&token_id)
                                && tok.masked
                            {
                                continue;
                            }
                            let score = (count as f32 / total_predictions as f32) * w;
                            local.push((token_id, score, w));
                        }
                    }
                }
                local
            })
            .collect();

        // 4. Transitions in parallel (including relational traversal)
        let edge_candidates: Vec<&BrainEdge> =
            self.edges.values().filter(|edge| !edge.masked).collect();
        let last_token = context.last().copied();
        let transition_results: Vec<(u64, f32)> = if let Some(last_tok) = last_token {
            // Find concepts last_tok belongs to
            let mut concepts = Vec::new();
            for edge in &edge_candidates {
                if edge.kind == EdgeKind::ConceptMember && edge.source == last_tok {
                    concepts.push(edge.target);
                }
            }

            // Find synonyms (other members of those concepts)
            let mut synonyms = BTreeSet::new();
            for concept_id in concepts {
                for edge in &edge_candidates {
                    if edge.kind == EdgeKind::ConceptMember
                        && edge.target == concept_id
                        && edge.source != last_tok
                    {
                        synonyms.insert(edge.source);
                    }
                }
            }

            edge_candidates
                .par_iter()
                .filter_map(|edge| {
                    if edge.kind == EdgeKind::Transition {
                        if edge.source == last_tok {
                            Some((edge.target, edge.strength.clamp(0.0, 1.0)))
                        } else if synonyms.contains(&edge.source) {
                            // Relational traversal: apply relational decay factor (0.3)
                            Some((edge.target, (edge.strength * 0.3).clamp(0.0, 1.0)))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            Vec::new()
        };

        // Combine scores
        let mut candidate_map: BTreeMap<u64, ComponentScores> = BTreeMap::new();

        for (token_id, score, w) in context_results {
            let entry = candidate_map.entry(token_id).or_default();
            entry.context_score += score;
            entry.context_weight_sum += w;
            entry.sources.insert(CandidateSource::ContextPattern);
        }

        let sum_transition: f32 = transition_results.iter().map(|(_, s)| s).sum();
        for (token_id, strength) in transition_results {
            let entry = candidate_map.entry(token_id).or_default();
            entry.transition_score += if sum_transition > 0.0 {
                strength / sum_transition
            } else {
                0.0
            };
            entry.sources.insert(CandidateSource::TransitionEdge);
        }

        let sum_recent: f32 = recent_memory_results.iter().map(|(_, s)| s).sum();
        for (token_id, weight) in recent_memory_results {
            let entry = candidate_map.entry(token_id).or_default();
            entry.recent_memory_score += if sum_recent > 0.0 {
                weight / sum_recent
            } else {
                0.0
            };
            entry.sources.insert(CandidateSource::RecentMemory);
        }

        let sum_soft: f32 = soft_recall_results.iter().map(|(_, s)| s).sum();
        for (token_id, sim) in soft_recall_results {
            let entry = candidate_map.entry(token_id).or_default();
            entry.soft_recall_score += if sum_soft > 0.0 { sim / sum_soft } else { 0.0 };
            entry.sources.insert(CandidateSource::SoftRecall);
        }

        let mut list = Vec::new();
        for (token_id, entry) in candidate_map {
            if let Some(tok) = self.tokenizer.entries.get(&token_id)
                && tok.masked
            {
                continue;
            }

            let ctx_score = if entry.context_weight_sum > 0.0 {
                entry.context_score / entry.context_weight_sum
            } else {
                0.0
            };

            let score = ctx_score * gen_config.context_weight
                + entry.transition_score * gen_config.transition_weight
                + entry.recent_memory_score * gen_config.recent_memory_weight
                + entry.soft_recall_score * gen_config.soft_recall_weight;

            if score < gen_config.min_confidence {
                continue;
            }

            // Apply repetition penalty
            let occurrence = generated.iter().filter(|&&t| t == token_id).count();
            let penalized_score = if occurrence > 0 {
                score / gen_config.repetition_penalty.powi(occurrence as i32)
            } else {
                score
            };

            let source = if entry.sources.len() > 1 {
                CandidateSource::Mixed
            } else {
                entry
                    .sources
                    .into_iter()
                    .next()
                    .unwrap_or(CandidateSource::Mixed)
            };

            let occurrences = self
                .nodes
                .get(&token_id)
                .map(|n| n.activation_count)
                .unwrap_or(0);
            list.push(TokenCandidate {
                token_id,
                score: penalized_score,
                probability: 0.0,
                occurrences,
                source,
            });
        }

        // Apply softmax or temperature scaling
        if list.is_empty() {
            return list;
        }

        if gen_config.temperature <= 0.01 {
            // Greedy
            list.sort_by(|a, b| b.score.total_cmp(&a.score));
            list[0].probability = 1.0;
            list.truncate(1);
        } else {
            // Normal softmax with temperature
            let mut sum_exp = 0.0f32;
            let mut exps = Vec::with_capacity(list.len());
            for c in &list {
                let exp = (c.score / gen_config.temperature).exp();
                exps.push(exp);
                sum_exp += exp;
            }
            if sum_exp > 0.0 {
                for (c, exp) in list.iter_mut().zip(exps) {
                    c.probability = exp / sum_exp;
                }
            } else {
                let size = list.len() as f32;
                for c in &mut list {
                    c.probability = 1.0 / size;
                }
            }
            // Sort by probability descending
            list.sort_by(|a, b| b.probability.total_cmp(&a.probability));
        }

        list
    }

    pub fn sample_token(
        &self,
        candidates: &[TokenCandidate],
        rng: &mut SimpleRng,
        gen_config: &GenerationConfig,
    ) -> Option<u64> {
        if candidates.is_empty() {
            return None;
        }

        // Top-k filtering
        let k = gen_config.top_k.max(1);
        let mut top_k_candidates = candidates.to_vec();
        if top_k_candidates.len() > k {
            top_k_candidates.truncate(k);
        }

        // Re-normalize probabilities
        let total_p: f32 = top_k_candidates.iter().map(|c| c.probability).sum();
        if total_p <= 0.0 {
            return Some(top_k_candidates[0].token_id);
        }
        for c in &mut top_k_candidates {
            c.probability /= total_p;
        }

        // Top-p (nucleus) filtering
        let mut cumulative_p = 0.0;
        let mut top_p_candidates = Vec::new();
        for c in top_k_candidates {
            cumulative_p += c.probability;
            top_p_candidates.push(c);
            if cumulative_p >= gen_config.top_p {
                break;
            }
        }

        // Re-normalize top-p probabilities
        let total_p: f32 = top_p_candidates.iter().map(|c| c.probability).sum();
        if total_p <= 0.0 {
            return Some(top_p_candidates[0].token_id);
        }
        for c in &mut top_p_candidates {
            c.probability /= total_p;
        }

        // Probabilistic sampling
        let mut r = rng.next_f32() * total_p;
        for c in &top_p_candidates {
            if r < c.probability {
                return Some(c.token_id);
            }
            r -= c.probability;
        }

        Some(top_p_candidates[0].token_id)
    }

    pub fn expand_context_tokens(&self, token_ids: &[u64]) -> Vec<u64> {
        let mut expanded = Vec::new();
        for token_id in token_ids {
            self.expand_token_into(*token_id, &mut expanded, 0);
        }
        expanded
    }

    pub fn expand_token_into(&self, token_id: u64, output: &mut Vec<u64>, depth: usize) {
        if depth >= 4 {
            output.push(token_id);
            return;
        }

        let node = match self.nodes.get(&token_id) {
            Some(n) => n,
            None => {
                output.push(token_id);
                return;
            }
        };

        if node.kind != NodeKind::Phrase {
            output.push(token_id);
            return;
        }

        for child in &node.composition {
            self.expand_token_into(*child, output, depth + 1);
        }
    }

    pub fn describe_tokens(&self, token_ids: &[u64]) -> String {
        let surfaces: Vec<String> = token_ids
            .iter()
            .filter_map(|token_id| self.tokenizer.get(*token_id))
            .map(|token| token.text.clone())
            .collect();
        collapse_whitespace(&surfaces.join(" "))
    }

    pub fn node_label(&self, node_id: u64) -> String {
        self.nodes
            .get(&node_id)
            .map(|node| node.label.clone())
            .unwrap_or_else(|| format!("node#{node_id}"))
    }
}

fn seed_with_offset(seed: Option<u64>, offset: usize) -> Option<u64> {
    seed.map(|value| value.wrapping_add(offset as u64))
}

fn unique_procedures(candidates: &[ResponseActionCandidate]) -> Vec<String> {
    let mut procedures = BTreeSet::new();
    for procedure_name in candidates
        .iter()
        .filter_map(|candidate| candidate.procedure_name.clone())
    {
        procedures.insert(procedure_name);
    }
    procedures.into_iter().take(3).collect()
}

fn infer_intents(prompt_text: &str) -> Vec<String> {
    let mut intents = Vec::new();
    let trimmed = prompt_text.trim();

    if trimmed.ends_with('?')
        || trimmed.starts_with("apa")
        || trimmed.starts_with("siapa")
        || trimmed.starts_with("bagaimana")
        || trimmed.starts_with("mengapa")
    {
        intents.push("question".to_string());
    }
    if trimmed.contains("tolong")
        || trimmed.starts_with("buat")
        || trimmed.starts_with("jelaskan")
        || trimmed.starts_with("ubah")
    {
        intents.push("instruction".to_string());
    }
    if trimmed.contains("saya") || trimmed.contains("aku") {
        intents.push("self-disclosure".to_string());
    }
    if intents.is_empty() {
        intents.push("statement".to_string());
    }

    intents
}

fn infer_goals(prompt_text: &str, intents: &[String]) -> Vec<String> {
    let mut goals = Vec::new();
    for intent in intents {
        match intent.as_str() {
            "question" => goals.push(format!("jawab pertanyaan: {prompt_text}")),
            "instruction" => goals.push(format!("selesaikan instruksi: {prompt_text}")),
            _ => {}
        }
    }
    goals.sort();
    goals.dedup();
    goals
}

fn infer_emotional_tone(prompt_text: &str) -> String {
    let lowered = prompt_text.to_lowercase();
    if lowered.contains("tolong") || lowered.contains("bantu") {
        "requesting".to_string()
    } else if lowered.contains("bingung") || lowered.contains("sulit") {
        "uncertain".to_string()
    } else if lowered.contains("terima kasih") || lowered.contains("senang") {
        "positive".to_string()
    } else if lowered.contains("marah") || lowered.contains("kesal") {
        "frustrated".to_string()
    } else {
        "neutral".to_string()
    }
}

fn estimate_prediction_error(
    prompt_text: &str,
    prompt_token_ids: &[u64],
    candidates: &[ResponseActionCandidate],
    chosen: &ResponseActionCandidate,
    predicted_procedures: &[String],
) -> f32 {
    let alignment =
        ((chosen.context_match + chosen.semantic_match + chosen.confidence) / 3.0).clamp(0.0, 1.0);
    let base_mismatch = 1.0 - alignment;
    let ambiguity = if candidates.len() > 1 {
        let top_score = chosen.score.max(0.0001);
        let delta = (chosen.score - candidates[1].score).max(0.0);
        1.0 - (delta / top_score).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let procedural_expectation = if let Some(query) = parse_arithmetic_query(prompt_text) {
        let expected = procedure_name(query.kind);
        if chosen.procedure_name.as_deref() == Some(expected) {
            0.0
        } else {
            0.35 + if predicted_procedures.iter().any(|name| name == expected) {
                0.15
            } else {
                0.0
            }
        }
    } else {
        0.0
    };
    let sparse_context_penalty = if prompt_token_ids.len() <= 1 {
        0.15
    } else {
        0.0
    };

    (base_mismatch * 0.55
        + ambiguity * 0.2
        + procedural_expectation * 0.2
        + sparse_context_penalty * 0.05)
        .clamp(0.0, 1.0)
}

fn format_procedural_response(prompt_text: &str, result: i64) -> String {
    if prompt_text.contains("berapa") || prompt_text.ends_with('?') {
        format!("hasilnya {result}.")
    } else {
        result.to_string()
    }
}

fn intrinsic_reward(candidate: &ResponseActionCandidate, prediction_error: f32) -> f32 {
    (candidate.context_match * 0.28
        + candidate.episodic_match * 0.18
        + candidate.semantic_match * 0.22
        + candidate.confidence * 0.22
        + candidate.novelty * 0.10
        - prediction_error.clamp(0.0, 1.0) * 0.18)
        .clamp(0.0, 1.0)
}
