use super::similarity::jaccard_similarity;
use super::state::BrainState;
use rayon::prelude::*;
use std::collections::BTreeMap;

#[derive(Clone, Debug)]
pub struct SimilarPrompt {
    pub prompt: String,
    pub token_ids: Vec<u64>,
    pub response: String,
    pub similarity: f32,
}

impl BrainState {
    pub fn find_similar_prompts(&self, prompt_tokens: &[u64]) -> Vec<SimilarPrompt> {
        let memories: Vec<(&String, &BTreeMap<String, u64>)> =
            self.prompt_response_memory.iter().collect();
        memories
            .par_iter()
            .filter_map(|(mem_prompt, responses)| {
                let mem_tokens = self.tokenizer.tokenize(mem_prompt);
                let sim = jaccard_similarity(prompt_tokens, &mem_tokens);
                if sim >= 0.25 {
                    // Get most frequent response
                    if let Some((response_text, _)) =
                        responses
                            .iter()
                            .max_by(|(left_r, left_c), (right_r, right_c)| {
                                left_c
                                    .cmp(right_c)
                                    .then_with(|| right_r.len().cmp(&left_r.len()))
                            })
                    {
                        return Some(SimilarPrompt {
                            prompt: (*mem_prompt).clone(),
                            token_ids: mem_tokens,
                            response: response_text.clone(),
                            similarity: sim,
                        });
                    }
                }
                None
            })
            .collect()
    }
}
