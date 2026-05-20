use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use rayon::prelude::*;

use super::config::GenerationConfig;
use super::error::BrainError;
use super::learning::{context_key, normalize_input, LearningReport};
use super::state::{BrainEdge, BrainState, EdgeKind, NodeKind, UtteranceMemory};
use super::tokenizer::{collapse_whitespace, TokenLevel};
use super::sampling::SimpleRng;
use super::similarity::jaccard_similarity;

#[derive(Clone, Debug)]
pub struct InteractionReport {
    pub learning: LearningReport,
    pub response: String,
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


impl BrainState {
    pub fn interact_with_seed(
        &mut self,
        input: &str,
        seed: Option<u64>,
        gen_config: &GenerationConfig,
    ) -> Result<InteractionReport, BrainError> {
        let learning = self.learn_text(input)?;
        self.interaction_count += 1; // Live interaction
        let response = self.generate_response_from_tokens_with_seed(&learning.token_ids, seed, gen_config)?;
        Ok(InteractionReport { learning, response })
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
        self.generate_response_from_tokens_with_seed(&token_ids, seed, gen_config)
    }

    pub fn generate_response(&self, prompt: &str) -> Result<String, BrainError> {
        self.generate_response_with_seed(prompt, None, &self.config.generation_config)
    }

    pub fn generate_response_from_tokens(&self, prompt_token_ids: &[u64]) -> Result<String, BrainError> {
        self.generate_response_from_tokens_with_seed(prompt_token_ids, None, &self.config.generation_config)
    }

    pub fn generate_response_from_tokens_with_seed(
        &self,
        prompt_token_ids: &[u64],
        seed: Option<u64>,
        gen_config: &GenerationConfig,
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
        let mut rng = SimpleRng::new(seed);

        for _ in 0..self.config.response_token_limit {
            let mut candidates = self.next_token_distribution(&context, prompt_token_ids, &generated, gen_config);

            // Trigram blocking: block any token that would repeat a trigram already present in `generated`
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
                        .filter(|c| !blocked_tokens.contains(&c.token_id))
                        .cloned()
                        .collect();
                    if !filtered.is_empty() {
                        candidates = filtered;
                        // Re-normalize probabilities
                        let total_p: f32 = candidates.iter().map(|c| c.probability).sum();
                        if total_p > 0.0 {
                            for c in &mut candidates {
                                c.probability /= total_p;
                            }
                        }
                    }
                }
            }

            if candidates.is_empty() {
                break;
            }

            // stop if top probability is lower than min_confidence
            if candidates[0].probability < gen_config.min_confidence {
                break;
            }

            let next_token = match self.sample_token(&candidates, &mut rng, gen_config) {
                Some(tok) => tok,
                None => break,
            };

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
                    let key_tokens = self.tokenizer.tokenize(&key_prompt);
                    let sim = jaccard_similarity(&prompt_tokens, &key_tokens);
                    if sim >= 0.25 {
                        if let Some((response_text, _)) = responses.iter().max_by(|(left_r, left_c), (right_r, right_c)| {
                            left_c.cmp(right_c).then_with(|| right_r.len().cmp(&left_r.len()))
                        }) {
                            let response_tokens = self.tokenizer.tokenize(response_text);
                            if g_len < response_tokens.len() {
                                return Some((response_tokens[g_len], sim));
                            }
                        }
                    }
                }
                None
            })
            .collect();

        // 2. Recent memory in parallel
        let utterances: Vec<&UtteranceMemory> = self.recent_utterances.iter().collect();
        let recent_memory_results: Vec<(u64, f32)> = utterances.par_iter().enumerate().flat_map(|(idx, utterance)| {
            let mut local = Vec::new();
            let n = utterances.len();
            let recency_weight = (idx + 1) as f32 / n as f32;
            if let Some(&last_token) = context.last() {
                for i in 0..utterance.token_ids.len() {
                    if utterance.token_ids[i] == last_token && i + 1 < utterance.token_ids.len() {
                        local.push((utterance.token_ids[i + 1], recency_weight));
                    }
                }
            }
            for &t in &utterance.token_ids {
                local.push((t, 0.05 * recency_weight));
            }
            local
        }).collect();

        // 3. Context pattern in parallel
        let max_window = self.config.max_context_window.min(context.len());
        let windows: Vec<Vec<u64>> = (1..=max_window)
            .map(|w| context[context.len() - w..].to_vec())
            .collect();
        let context_results: Vec<(u64, f32, f32)> = windows.par_iter().enumerate().flat_map(|(index, prefix)| {
            let w = (index + 1) as f32;
            let mut local = Vec::new();
            if let Some(pattern) = self.context_patterns.get(&context_key(prefix)) {
                let total_predictions: u64 = pattern.predicted_counts.values().sum();
                if total_predictions > 0 {
                    for (&token_id, &count) in &pattern.predicted_counts {
                        let score = (count as f32 / total_predictions as f32) * w;
                        local.push((token_id, score, w));
                    }
                }
            }
            local
        }).collect();

        // 4. Transitions in parallel (including relational traversal)
        let edge_candidates: Vec<&BrainEdge> = self.edges.values().collect();
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
                    if edge.kind == EdgeKind::ConceptMember && edge.target == concept_id && edge.source != last_tok {
                        synonyms.insert(edge.source);
                    }
                }
            }

            edge_candidates.par_iter().filter_map(|edge| {
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
            }).collect()
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
            entry.transition_score += if sum_transition > 0.0 { strength / sum_transition } else { 0.0 };
            entry.sources.insert(CandidateSource::TransitionEdge);
        }

        let sum_recent: f32 = recent_memory_results.iter().map(|(_, s)| s).sum();
        for (token_id, weight) in recent_memory_results {
            let entry = candidate_map.entry(token_id).or_default();
            entry.recent_memory_score += if sum_recent > 0.0 { weight / sum_recent } else { 0.0 };
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
                entry.sources.into_iter().next().unwrap_or(CandidateSource::Mixed)
            };

            let occurrences = self.nodes.get(&token_id).map(|n| n.activation_count).unwrap_or(0);
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
