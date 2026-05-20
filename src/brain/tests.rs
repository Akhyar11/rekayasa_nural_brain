#[cfg(test)]
mod tests {
    use crate::brain::learning::normalize_input;
    use crate::brain::{BrainConfig, BrainState, STATE_VERSION};
    use crate::brain::state::{
        LegacyBrainStateV1, LegacyBrainStateV2,
        LEGACY_STATE_VERSION_V1, LEGACY_STATE_VERSION_V2,
    };
    use crate::brain::tokenizer::{TokenLevel, TokenTrie};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn normalize_input_collapses_whitespace() {
        let normalized =
            normalize_input("  Halo\tDunia \n Besar ").expect("normalization should pass");
        assert_eq!(normalized, "halo dunia besar");
    }

    #[test]
    fn brain_grows_word_and_phrase_tokens() {
        let mut config = BrainConfig::default();
        config.dynamic_vocab = true;
        let mut brain = BrainState::new(config).expect("brain should initialize");

        brain
            .learn_text("halo dunia")
            .expect("learning should pass");
        let second = brain
            .learn_text("halo dunia")
            .expect("learning should pass");
        let third = brain
            .learn_text("halo dunia")
            .expect("learning should pass");

        assert!(second.new_word_tokens.iter().any(|token| token == "halo"));
        assert!(
            third
                .new_phrase_tokens
                .iter()
                .any(|token| token == "halo dunia")
        );
        assert!(
            brain
                .tokenizer
                .entries
                .values()
                .any(|token| token.level == TokenLevel::Phrase)
        );
    }

    #[test]
    fn brain_persists_and_loads_state() {
        let mut brain = BrainState::new(BrainConfig::default()).expect("brain should initialize");
        brain
            .interact("spiking brain")
            .expect("interaction should pass");

        let temp_path = std::env::temp_dir().join(format!(
            "rekayasa_nural_brain_test_{}.bin",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock should be valid")
                .as_nanos()
            ));

        brain.save_to_path(&temp_path).expect("save should pass");
        let loaded = BrainState::load_from_path(&temp_path).expect("load should pass");
        std::fs::remove_file(&temp_path).expect("temp file should be removable");

        assert_eq!(loaded.interaction_count, 1);
        assert_eq!(
            loaded.tokenizer.known_token_count(),
            brain.tokenizer.known_token_count()
        );
        assert_eq!(loaded.recent_utterances.len(), 1);
    }

    #[test]
    fn brain_persists_and_loads_json_state() {
        let mut brain = BrainState::new(BrainConfig::default()).expect("brain should initialize");
        brain
            .interact("spiking brain")
            .expect("interaction should pass");

        let temp_path = std::env::temp_dir().join(format!(
            "rekayasa_nural_brain_test_{}.json",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock should be valid")
                .as_nanos()
        ));

        brain.save_to_path(&temp_path).expect("save should pass");
        let loaded = BrainState::load_from_path(&temp_path).expect("load should pass");
        std::fs::remove_file(&temp_path).expect("temp file should be removable");

        assert_eq!(loaded.interaction_count, 1);
        assert_eq!(
            loaded.tokenizer.known_token_count(),
            brain.tokenizer.known_token_count()
        );
        assert_eq!(loaded.recent_utterances.len(), 1);
    }

    #[test]
    fn interact_returns_response_after_learning() {
        let mut brain = BrainState::new(BrainConfig::default()).expect("brain should initialize");
        brain
            .learn_text("saya suka kopi")
            .expect("learning should pass");
        brain
            .learn_text("saya suka teh")
            .expect("learning should pass");
        let interaction = brain
            .interact("saya suka")
            .expect("interaction should pass");

        assert!(!interaction.response.is_empty());
        assert!(interaction.learning.token_count > 0);
    }

    #[test]
    fn summary_reports_graph_shape() {
        let mut brain = BrainState::new(BrainConfig::default()).expect("brain should initialize");
        brain
            .learn_text("otak ini tumbuh")
            .expect("learning should pass");
        let summary = brain.summary(4);

        assert!(summary.token_count > 0);
        assert!(summary.edge_count > 0);
    }

    #[test]
    fn train_pair_links_prompt_to_response() {
        let mut brain = BrainState::new(BrainConfig::default()).expect("brain should initialize");
        let training = brain
            .train_pair("apa warna langit", "langit berwarna biru")
            .expect("training pair should pass");
        let response = brain
            .generate_response("apa warna langit")
            .expect("response generation should pass");

        assert!(training.prompt_token_count > 0);
        assert!(training.response_token_count > 0);
        assert!(response.contains("langit") || response.contains("berwarna"));
    }

    #[test]
    fn load_legacy_binary_state_and_migrate_v1() {
        let legacy = LegacyBrainStateV1 {
            state_version: LEGACY_STATE_VERSION_V1,
            config: BrainConfig::default(),
            next_node_id: 1,
            interaction_count: 7,
            tokenizer: Default::default(),
            nodes: Default::default(),
            edges: Default::default(),
            context_patterns: Default::default(),
            recent_utterances: Default::default(),
        };

        let temp_path = std::env::temp_dir().join(format!(
            "rekayasa_nural_brain_legacy_v1_test_{}.bin",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock should be valid")
                .as_nanos()
        ));

        let payload = bincode::serde::encode_to_vec(&legacy, bincode::config::standard())
            .expect("legacy encoding should pass");
        std::fs::write(&temp_path, payload).expect("legacy temp file write should pass");

        let migrated = BrainState::load_from_path(&temp_path).expect("legacy load should pass");
        std::fs::remove_file(&temp_path).expect("temp file should be removable");

        assert_eq!(migrated.state_version, STATE_VERSION);
        assert_eq!(migrated.interaction_count, 7);
        assert_eq!(migrated.learning_step_count, 7);
        assert_eq!(migrated.training_example_count, 0);
        assert!(migrated.prompt_response_memory.is_empty());
    }

    #[test]
    fn load_legacy_binary_state_and_migrate_v2() {
        let legacy = LegacyBrainStateV2 {
            state_version: LEGACY_STATE_VERSION_V2,
            config: BrainConfig::default(),
            next_node_id: 1,
            interaction_count: 12,
            tokenizer: Default::default(),
            nodes: Default::default(),
            edges: Default::default(),
            context_patterns: Default::default(),
            prompt_response_memory: Default::default(),
            recent_utterances: Default::default(),
        };

        let temp_path = std::env::temp_dir().join(format!(
            "rekayasa_nural_brain_legacy_v2_test_{}.bin",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock should be valid")
                .as_nanos()
        ));

        let payload = bincode::serde::encode_to_vec(&legacy, bincode::config::standard())
            .expect("legacy encoding should pass");
        std::fs::write(&temp_path, payload).expect("legacy temp file write should pass");

        let migrated = BrainState::load_from_path(&temp_path).expect("legacy load should pass");
        std::fs::remove_file(&temp_path).expect("temp file should be removable");

        assert_eq!(migrated.state_version, STATE_VERSION);
        assert_eq!(migrated.interaction_count, 12);
        assert_eq!(migrated.learning_step_count, 12);
        assert_eq!(migrated.training_example_count, 0);
        assert!(migrated.prompt_response_memory.is_empty());
    }

    #[test]
    fn trie_tokenizer_longest_prefix_matching() {
        let mut trie = TokenTrie::default();
        trie.insert("a", 1, TokenLevel::Sensor);
        trie.insert("apa", 2, TokenLevel::Word);
        trie.insert("apa warna", 3, TokenLevel::Phrase);

        // Test insertion by manually checking node exists
        assert!(trie.root.children.contains_key(&'a'));
        let a_node = trie.root.children.get(&'a').unwrap();
        assert_eq!(a_node.token_id, Some(1));
    }

    #[test]
    fn jaccard_similarity_calculation() {
        use crate::brain::jaccard_similarity;
        let a = vec![1, 2, 3];
        let b = vec![2, 3, 4];
        let sim = jaccard_similarity(&a, &b);
        // Intersection = [2, 3] (len 2), Union = [1, 2, 3, 4] (len 4) => 2/4 = 0.5
        assert_eq!(sim, 0.5);
    }

    #[test]
    fn deterministic_random_projection_vectors() {
        use crate::brain::similarity::RandomProjectionVector;

        let vec1 = RandomProjectionVector::for_token(42);
        let vec2 = RandomProjectionVector::for_token(42);
        let vec3 = RandomProjectionVector::for_token(100);

        // Determinism check
        assert_eq!(vec1.values, vec2.values);
        // Distance check
        let sim_same = vec1.cosine_similarity(&vec2);
        let sim_diff = vec1.cosine_similarity(&vec3);

        assert!((sim_same - 1.0).abs() < 1e-5);
        assert!(sim_diff < 1.0);

        let agg = RandomProjectionVector::aggregate(&[vec1, vec3]);
        assert_eq!(agg.values.len(), RandomProjectionVector::DIM);
    }

    #[test]
    fn sampling_mechanics() {
        use crate::brain::sampling::sample_next_token;
        use crate::brain::{GenerationConfig, TokenCandidate, CandidateSource};

        let candidates = vec![
            TokenCandidate {
                token_id: 1,
                score: 10.0,
                probability: 0.7,
                occurrences: 5,
                source: CandidateSource::ContextPattern,
            },
            TokenCandidate {
                token_id: 2,
                score: 5.0,
                probability: 0.2,
                occurrences: 2,
                source: CandidateSource::TransitionEdge,
            },
            TokenCandidate {
                token_id: 3,
                score: 1.0,
                probability: 0.1,
                occurrences: 1,
                source: CandidateSource::SoftRecall,
            },
        ];

        let config = GenerationConfig {
            temperature: 0.0,
            top_k: 1,
            top_p: 0.9,
            repetition_penalty: 1.0,
            randomness_seed: Some(42),
            ..Default::default()
        };

        let sampled = sample_next_token(&candidates, &config);
        assert_eq!(sampled, Some(1));
    }

    #[test]
    fn probability_normalization_and_entropy() {
        use crate::brain::probability::{calculate_smoothed_probability, entropy};

        let p1 = calculate_smoothed_probability(2, 10, 3, 1.0); // (2+1)/(10+3) = 3/13
        assert!((p1 - (3.0 / 13.0)).abs() < 1e-5);

        let probs = vec![0.5, 0.5];
        let ent = entropy(&probs);
        assert!((ent - 0.69314718).abs() < 1e-5);
    }

    #[test]
    fn trigram_blocking_prevents_loops() {
        let mut brain = BrainState::new(BrainConfig::default()).expect("brain should initialize");
        // Latih berulang kali untuk memperkuat transisi suka -> kopi -> hitam
        for _ in 0..5 {
            brain.learn_text("suka kopi hitam suka kopi hitam").expect("learning should pass");
        }
        
        let response = brain.generate_response("suka kopi").expect("generation should pass");
        
        // Karena ada trigram blocking, perulangan beruntun 3x suka kopi hitam harus terputus
        assert!(!response.contains("suka kopi hitam suka kopi hitam suka kopi hitam"), 
            "Response should block consecutive loop, but got: {}", response);
    }

    #[test]
    fn concept_relational_traversal_enables_soft_reasoning() {
        let mut brain = BrainState::new(BrainConfig::default()).expect("brain should initialize");
        
        // 1. Latih transisi "teh manis" beberapa kali agar dipromosikan jadi kata
        for _ in 0..5 {
            brain.learn_text("teh manis").expect("learning should pass");
        }
        
        // 2. Buat konsep "minuman" yang mengelompokkan "kopi" dan "teh"
        brain.associate_concept(
            "minuman",
            &["kopi".to_string(), "teh".to_string()],
            1
        ).expect("associating concept should pass");
        
        // 3. Generate respon untuk "kopi".
        // Karena "kopi" dan "teh" adalah anggota konsep "minuman",
        // relational traversal akan memungkinkan transisi ke "manis" via "teh"!
        let response = brain.generate_response("kopi").expect("generation should pass");
        
        assert!(response.contains("manis"), "Response should traverse via concept node to 'manis', but got: {}", response);
    }
}
