use super::state::{BrainState, EdgeKind, NodeKind};
use super::tokenizer::TokenLevel;
use std::collections::HashSet;

impl BrainState {
    pub fn prune_graph(&mut self, interaction_index: u64) -> (usize, usize) {
        let stale_after = self.config.prune_interval.saturating_mul(4);

        // 1. Kumpulkan token (kata/frasa) yang sudah usang
        let mut tokens_to_mask = HashSet::new();
        for (&node_id, entry) in &self.tokenizer.entries {
            // Jangan pernah hapus/mask token khusus sistem (diawali <) atau sensor dasar (huruf/angka tunggal)
            if entry.text.starts_with("<") || entry.level == TokenLevel::Sensor {
                continue;
            }

            // Cek usia keaktifan token
            let age = interaction_index.saturating_sub(entry.last_used_at);
            if age >= stale_after {
                tokens_to_mask.insert(node_id);
            }
        }

        // Mask token yang usang di tokenizer dan graf node
        let mut total_tokens_masked = 0;
        for &node_id in &tokens_to_mask {
            if let Some(entry) = self.tokenizer.entries.get_mut(&node_id)
                && !entry.masked
            {
                entry.masked = true;
                total_tokens_masked += 1;
            }
            if let Some(node) = self.nodes.get_mut(&node_id) {
                node.masked = true;
            }
        }

        // 2. Tentukan edges yang akan di-mask
        let mut edges_masked_count = 0;
        for edge in self.edges.values_mut() {
            // Edges yang terhubung ke token yang telah di-mask harus ikut di-mask (cascading masking)
            let is_connected_to_masked_token =
                tokens_to_mask.contains(&edge.source) || tokens_to_mask.contains(&edge.target);

            let age = interaction_index.saturating_sub(edge.last_activated_at);
            if edge.kind != EdgeKind::ConceptMember && age > 1 {
                edge.strength *= self.config.edge_decay;
            }

            let is_stale_weak = edge.kind != EdgeKind::ConceptMember
                && edge.strength < self.config.min_edge_strength
                && age >= stale_after;

            if (is_connected_to_masked_token || is_stale_weak) && !edge.masked {
                edge.masked = true;
                edges_masked_count += 1;
            }
        }

        // 3. Tentukan Context Patterns dan Nodes yang akan di-mask
        let mut context_nodes_masked_count = 0;
        for node in self.nodes.values_mut() {
            if node.kind != NodeKind::Context {
                continue;
            }
            let age = interaction_index.saturating_sub(node.last_activated_at);
            if age >= stale_after && !node.masked {
                node.masked = true;
                context_nodes_masked_count += 1;
            }
        }

        for pattern in self.context_patterns.values_mut() {
            let age = interaction_index.saturating_sub(pattern.last_activated_at);
            if age >= stale_after && !pattern.masked {
                pattern.masked = true;
            }
        }

        let total_nodes_masked = context_nodes_masked_count + total_tokens_masked;
        (edges_masked_count, total_nodes_masked)
    }
}
