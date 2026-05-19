use std::collections::BTreeSet;

use super::learning::{context_key, normalize_input, LearningReport};
use super::state::{BrainEdge, BrainError, BrainState, EdgeKind, NodeKind};
use super::tokenizer::{collapse_whitespace, token_priority, TokenLevel};

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
    pub learning_steps: u64,
    pub training_examples: u64,
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

struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new() -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        Self {
            state: seed ^ 0x5555555555555555,
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() & 0xFFFFFFFF) as f32 / 4294967296.0
    }
}

pub(crate) fn jaccard_similarity(a: &[u64], b: &[u64]) -> f32 {

    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let set_a: BTreeSet<u64> = a.iter().copied().collect();
    let set_b: BTreeSet<u64> = b.iter().copied().collect();
    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();
    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

impl BrainState {
    pub fn interact(&mut self, input: &str) -> Result<InteractionReport, BrainError> {
        let learning = self.learn_text(input)?;
        self.interaction_count += 1; // Live interaction
        let response = if let Some(response) = self.recall_trained_response(&learning.normalized_text) {
            response
        } else {
            self.generate_response_from_tokens(&learning.token_ids)?
        };
        Ok(InteractionReport { learning, response })
    }

    pub fn generate_response(&self, prompt: &str) -> Result<String, BrainError> {
        let normalized_text = normalize_input(prompt)?;
        if let Some(response) = self.recall_trained_response(&normalized_text) {
            return Ok(response);
        }
        let token_ids = self.tokenizer.tokenize(&normalized_text);
        self.generate_response_from_tokens(&token_ids)
    }

    pub fn generate_response_from_tokens(
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
        let mut rng = SimpleRng::new();

        for _ in 0..self.config.response_token_limit {
            let candidates = self.get_candidates(&context);
            if candidates.is_empty() {
                break;
            }

            // Repetition penalty
            let mut penalized_candidates: Vec<(u64, f32)> = candidates
                .into_iter()
                .map(|(token_id, weight)| {
                    let occurrence = generated.iter().filter(|&&t| t == token_id).count();
                    let penalty = 1.3f32.powi(occurrence as i32);
                    (token_id, weight / penalty)
                })
                .collect();

            // Sort by weight descending
            penalized_candidates.sort_by(|a, b| b.1.total_cmp(&a.1));

            // Top-k (k = 5)
            let top_k = 5;
            let top_candidates = if penalized_candidates.len() > top_k {
                &penalized_candidates[..top_k]
            } else {
                &penalized_candidates[..]
            };

            let total_weight: f32 = top_candidates.iter().map(|(_, w)| w).sum();
            if total_weight <= 0.0 {
                break;
            }

            // Probabilistic sampling
            let mut r = rng.next_f32() * total_weight;
            let mut next_token = top_candidates[0].0;
            for (token_id, weight) in top_candidates {
                if r < *weight {
                    next_token = *token_id;
                    break;
                }
                r -= *weight;
            }

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

    pub fn get_candidates(&self, context: &[u64]) -> Vec<(u64, f32)> {
        let max_window = self.config.max_context_window.min(context.len());
        // 1. Try matching context patterns
        for window_size in (1..=max_window).rev() {
            let prefix = &context[context.len() - window_size..];
            if let Some(pattern) = self.context_patterns.get(&context_key(prefix)) {
                let total_predictions: u64 = pattern.predicted_counts.values().sum();
                if total_predictions > 0 {
                    return pattern
                        .predicted_counts
                        .iter()
                        .map(|(&token_id, &count)| {
                            let priority = token_priority(
                                self.tokenizer
                                    .get(token_id)
                                    .map(|token| token.level)
                                    .unwrap_or(TokenLevel::Sensor),
                            );
                            let weight = (count as f32 / total_predictions as f32) + (priority as f32 * 0.05);
                            (token_id, weight)
                        })
                        .collect();
                }
            }
        }

        // 2. Fallback to transition edges
        if let Some(&last_token) = context.last() {
            let mut transition_candidates = Vec::new();
            for edge in self.edges.values() {
                if edge.kind == EdgeKind::Transition && edge.source == last_token {
                    transition_candidates.push((edge.target, edge.strength.clamp(0.0, 1.0)));
                }
            }
            if !transition_candidates.is_empty() {
                return transition_candidates;
            }
        }

        Vec::new()
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

    pub fn recall_trained_response(&self, prompt: &str) -> Option<String> {
        // 1. Exact recall
        if let Some(responses) = self.prompt_response_memory.get(prompt) {
            if let Some((response, _)) = responses.iter().max_by(
                |(left_response, left_count), (right_response, right_count)| {
                    left_count
                        .cmp(right_count)
                        .then_with(|| right_response.len().cmp(&left_response.len()))
                },
            ) {
                return Some(response.clone());
            }
        }

        // 2. Approximate recall using Jaccard Similarity on tokenized representations
        let input_tokens = self.tokenizer.tokenize(prompt);
        if input_tokens.is_empty() {
            return None;
        }

        let mut best_key = None;
        let mut best_similarity = 0.0f32;

        for key in self.prompt_response_memory.keys() {
            let key_tokens = self.tokenizer.tokenize(key);
            let similarity = jaccard_similarity(&input_tokens, &key_tokens);
            if similarity > best_similarity {
                best_similarity = similarity;
                best_key = Some(key);
            }
        }

        if best_similarity >= 0.55 {
            if let Some(key) = best_key {
                let responses = self.prompt_response_memory.get(key)?;
                if let Some((response, _)) = responses.iter().max_by(
                    |(left_response, left_count), (right_response, right_count)| {
                        left_count
                            .cmp(right_count)
                            .then_with(|| right_response.len().cmp(&left_response.len()))
                    },
                ) {
                    return Some(response.clone());
                }
            }
        }

        None
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

        let mut latest_tokens: Vec<&super::tokenizer::TokenEntry> = self.tokenizer.entries.values().collect();
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
            latest_tokens: latest_tokens
                .into_iter()
                .take(8)
                .map(|token| token.text.clone())
                .collect(),
            strongest_edges,
        }
    }
}
