use std::collections::{BTreeMap, BTreeSet};

use super::error::BrainError;
use super::state::{
    BrainEdge, BrainNode, BrainState, EdgeKind, NodeKind,
    ContextPattern, UtteranceMemory,
};
use super::tokenizer::{collapse_whitespace, TokenLevel};

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
pub struct TrainingExampleReport {
    pub prompt: String,
    pub response: String,
    pub prompt_token_count: usize,
    pub response_token_count: usize,
    pub new_sensor_tokens: Vec<String>,
    pub new_word_tokens: Vec<String>,
    pub new_phrase_tokens: Vec<String>,
    pub new_context_nodes: Vec<String>,
    pub new_edges: usize,
    pub pruned_edges: usize,
    pub pruned_context_nodes: usize,
}

impl BrainState {
    pub fn learn_text(&mut self, input: &str) -> Result<LearningReport, BrainError> {
        self.learn_text_internal(input, true)
    }

    pub fn train_pair(
        &mut self,
        prompt: &str,
        response: &str,
    ) -> Result<TrainingExampleReport, BrainError> {
        let prompt_learning = self.learn_text_internal(prompt, false)?;
        let response_learning = self.learn_text_internal(response, false)?;

        let prompt_text = prompt_learning.normalized_text.clone();
        let response_text = response_learning.normalized_text.clone();

        self.learning_step_count += 1;
        let bridge_step = self.learning_step_count;
        let prompt_token_ids = self.tokenizer.tokenize(&prompt_text);
        let response_token_ids = self.tokenizer.tokenize(&response_text);

        if prompt_token_ids.is_empty() || response_token_ids.is_empty() {
            return Err(BrainError::EmptyInput);
        }

        let mut combined_token_ids = prompt_token_ids.clone();
        combined_token_ids.extend(response_token_ids.iter().copied());

        let mut bridge_learning = LearningReport {
            normalized_text: format!("{prompt_text} => {response_text}"),
            token_count: combined_token_ids.len(),
            new_sensor_tokens: Vec::new(),
            new_word_tokens: Vec::new(),
            new_phrase_tokens: Vec::new(),
            new_context_nodes: Vec::new(),
            new_edges: 0,
            pruned_edges: 0,
            pruned_context_nodes: 0,
            token_ids: combined_token_ids,
        };

        for token_id in &bridge_learning.token_ids {
            self.activate_node(*token_id, bridge_step)?;
            self.tokenizer.touch_token(*token_id, bridge_step)?;
        }

        bridge_learning.new_edges +=
            self.learn_transitions(&bridge_learning.token_ids, bridge_step);
        let bridge_token_ids = bridge_learning.token_ids.clone();
        bridge_learning.new_edges += self.learn_context_patterns(
            &bridge_token_ids,
            bridge_step,
            &mut bridge_learning,
        )?;

        if bridge_step.is_multiple_of(self.config.prune_interval) {
            let (pruned_edges, pruned_nodes) = self.prune_graph(bridge_step);
            bridge_learning.pruned_edges = pruned_edges;
            bridge_learning.pruned_context_nodes = pruned_nodes;
        }

        self.store_prompt_response_pair(&prompt_text, &response_text);
        self.training_example_count += 1;

        Ok(TrainingExampleReport {
            prompt: prompt_text,
            response: response_text,
            prompt_token_count: prompt_token_ids.len(),
            response_token_count: response_token_ids.len(),
            new_sensor_tokens: merge_unique_vectors(
                prompt_learning.new_sensor_tokens,
                response_learning.new_sensor_tokens,
            ),
            new_word_tokens: merge_unique_vectors(
                prompt_learning.new_word_tokens,
                response_learning.new_word_tokens,
            ),
            new_phrase_tokens: merge_unique_vectors(
                prompt_learning.new_phrase_tokens,
                response_learning.new_phrase_tokens,
            ),
            new_context_nodes: bridge_learning.new_context_nodes,
            new_edges: prompt_learning.new_edges
                + response_learning.new_edges
                + bridge_learning.new_edges,
            pruned_edges: prompt_learning.pruned_edges
                + response_learning.pruned_edges
                + bridge_learning.pruned_edges,
            pruned_context_nodes: prompt_learning.pruned_context_nodes
                + response_learning.pruned_context_nodes
                + bridge_learning.pruned_context_nodes,
        })
    }

    pub fn learn_text_internal(
        &mut self,
        input: &str,
        remember_utterance: bool,
    ) -> Result<LearningReport, BrainError> {
        self.config.validate()?;
        let normalized_text = normalize_input(input)?;
        self.learning_step_count += 1;
        let step_index = self.learning_step_count;
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

        self.ensure_sensor_tokens(&normalized_text, step_index, &mut report)?;
        self.promote_word_tokens(&normalized_text, step_index, &mut report)?;
        self.promote_phrase_tokens(&normalized_text, step_index, &mut report)?;

        let token_ids = self.tokenizer.tokenize(&normalized_text);
        report.token_count = token_ids.len();
        report.token_ids = token_ids.clone();
        if token_ids.is_empty() {
            return Err(BrainError::EmptyInput);
        }

        for token_id in &token_ids {
            self.activate_node(*token_id, step_index)?;
            self.tokenizer.touch_token(*token_id, step_index)?;
        }

        report.new_edges += self.learn_transitions(&token_ids, step_index);
        report.new_edges +=
            self.learn_context_patterns(&token_ids, step_index, &mut report)?;
        if remember_utterance {
            self.push_utterance(step_index, &normalized_text, token_ids);
        }

        if step_index.is_multiple_of(self.config.prune_interval) {
            let (pruned_edges, pruned_nodes) = self.prune_graph(step_index);
            report.pruned_edges = pruned_edges;
            report.pruned_context_nodes = pruned_nodes;
        }

        Ok(report)
    }

    pub fn ensure_sensor_tokens(
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

    pub fn promote_word_tokens(
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

    pub fn promote_phrase_tokens(
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

    pub fn create_token_node(
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

    pub fn create_context_node(
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

    pub fn learn_transitions(&mut self, token_ids: &[u64], interaction_index: u64) -> usize {
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

    pub fn learn_context_patterns(
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

    pub fn push_utterance(&mut self, interaction_index: u64, text: &str, token_ids: Vec<u64>) {
        if self.recent_utterances.len() >= self.config.max_recent_utterances {
            self.recent_utterances.pop_front();
        }
        self.recent_utterances.push_back(UtteranceMemory {
            interaction_index,
            text: text.to_string(),
            token_ids,
        });
    }


    pub fn strengthen_edge(
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

    pub fn activate_node(&mut self, node_id: u64, interaction_index: u64) -> Result<(), BrainError> {
        let node = self
            .nodes
            .get_mut(&node_id)
            .ok_or(BrainError::MissingNode(node_id))?;
        node.activation_count += 1;
        node.last_activated_at = interaction_index;
        node.salience = (node.salience + 0.2).clamp(0.0, 1.0);
        Ok(())
    }

    pub fn allocate_node_id(&mut self) -> u64 {
        let node_id = self.next_node_id;
        self.next_node_id += 1;
        node_id
    }

    pub fn store_prompt_response_pair(&mut self, prompt: &str, response: &str) {
        let responses = self
            .prompt_response_memory
            .entry(prompt.to_string())
            .or_default();
        *responses.entry(response.to_string()).or_insert(0) += 1;
    }
}

pub fn normalize_input(text: &str) -> Result<String, BrainError> {
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

pub fn merge_unique_vectors(left: Vec<String>, right: Vec<String>) -> Vec<String> {
    let mut merged = BTreeSet::new();
    merged.extend(left);
    merged.extend(right);
    merged.into_iter().collect()
}

pub fn context_key(token_ids: &[u64]) -> String {
    token_ids
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(":")
}

pub fn edge_key(source: u64, target: u64, kind: EdgeKind) -> String {
    format!("{source}:{target}:{kind:?}")
}
