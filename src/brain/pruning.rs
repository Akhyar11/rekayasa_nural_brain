use std::collections::BTreeSet;
use super::state::{BrainState, EdgeKind, NodeKind};
use super::tokenizer::TokenLevel;

impl BrainState {
    pub fn prune_graph(&mut self, interaction_index: u64) -> (usize, usize) {
        let stale_after = self.config.prune_interval.saturating_mul(4);

        // 1. Kumpulkan token (kata/frasa) yang sudah usang
        let mut tokens_to_remove = Vec::new();
        for (&node_id, entry) in &self.tokenizer.entries {
            // Jangan pernah hapus token khusus sistem (diawali <) atau sensor dasar (huruf/angka tunggal)
            if entry.text.starts_with("<") || entry.level == TokenLevel::Sensor {
                continue;
            }

            // Cek usia keaktifan token
            let age = interaction_index.saturating_sub(entry.last_used_at);
            if age >= stale_after {
                tokens_to_remove.push((node_id, entry.text.clone()));
            }
        }

        // Hapus token yang usang dari tokenizer dan node graf
        for (node_id, text) in &tokens_to_remove {
            self.tokenizer.entries.remove(node_id);
            self.tokenizer.lookup.remove(text);
            self.nodes.remove(node_id);
        }

        // Rebuild pencarian Trie BPE jika ada kosakata yang terhapus
        if !tokens_to_remove.is_empty() {
            self.tokenizer.rebuild_trie();
        }

        // 2. Tentukan edges yang akan dihapus
        let mut edges_to_remove = Vec::new();
        for (key, edge) in &mut self.edges {
            // Edges yang terhubung ke token yang telah dihapus harus ikut dihapus (cascading pruning)
            let is_connected_to_pruned_token = tokens_to_remove
                .iter()
                .any(|(id, _)| edge.source == *id || edge.target == *id);

            let age = interaction_index.saturating_sub(edge.last_activated_at);
            if edge.kind != EdgeKind::ConceptMember && age > 1 {
                edge.strength *= self.config.edge_decay;
            }

            let is_stale_weak = edge.kind != EdgeKind::ConceptMember
                && edge.strength < self.config.min_edge_strength
                && age >= stale_after;

            if is_connected_to_pruned_token || is_stale_weak {
                edges_to_remove.push(key.clone());
            }
        }

        for key in &edges_to_remove {
            self.edges.remove(key);
        }

        // 3. Tentukan Context Nodes yang akan dihapus
        let active_nodes: BTreeSet<u64> = self
            .edges
            .values()
            .flat_map(|edge| [edge.source, edge.target])
            .collect();
        let protected_token_nodes: BTreeSet<u64> = self.tokenizer.entries.keys().copied().collect();
        let mut context_nodes_to_remove = Vec::new();

        for (node_id, node) in &self.nodes {
            if node.kind != NodeKind::Context {
                continue;
            }
            if protected_token_nodes.contains(node_id) {
                continue;
            }
            let age = interaction_index.saturating_sub(node.last_activated_at);
            if !active_nodes.contains(node_id) && age >= stale_after {
                context_nodes_to_remove.push(*node_id);
            }
        }

        for node_id in &context_nodes_to_remove {
            self.nodes.remove(node_id);
        }

        for pattern in self.context_patterns.values_mut() {
            if let Some(node_id) = pattern.node_id
                && context_nodes_to_remove.contains(&node_id)
            {
                pattern.node_id = None;
            }
        }

        let total_nodes_removed = context_nodes_to_remove.len() + tokens_to_remove.len();
        (edges_to_remove.len(), total_nodes_removed)
    }
}
