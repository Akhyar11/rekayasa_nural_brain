use std::collections::BTreeSet;
use super::state::{BrainState, EdgeKind, NodeKind};

impl BrainState {
    pub fn prune_graph(&mut self, interaction_index: u64) -> (usize, usize) {
        let stale_after = self.config.prune_interval.saturating_mul(4);
        let mut edges_to_remove = Vec::new();
        for (key, edge) in &mut self.edges {
            let age = interaction_index.saturating_sub(edge.last_activated_at);
            if edge.kind != EdgeKind::ConceptMember && age > 1 {
                edge.strength *= self.config.edge_decay;
            }
            if edge.kind != EdgeKind::ConceptMember
                && edge.strength < self.config.min_edge_strength
                && age >= stale_after
            {
                edges_to_remove.push(key.clone());
            }
        }

        for key in &edges_to_remove {
            self.edges.remove(key);
        }

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

        (edges_to_remove.len(), context_nodes_to_remove.len())
    }
}
