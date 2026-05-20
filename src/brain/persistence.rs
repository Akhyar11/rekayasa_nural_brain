use std::fs;
use std::path::Path;

use super::config::BrainConfig;
use super::error::BrainError;
use super::state::{
    BrainState, LEGACY_STATE_VERSION_V1, LEGACY_STATE_VERSION_V2, LEGACY_STATE_VERSION_V3,
    LegacyBrainStateV1, LegacyBrainStateV2, LegacyBrainStateV3, STATE_VERSION,
    temporary_state_path,
};

impl BrainState {
    pub fn load_or_new(path: &Path, config: BrainConfig) -> Result<Self, BrainError> {
        if path.exists() {
            let mut state = Self::load_from_path(path)?;
            config.validate()?;
            state.config = config;
            Ok(state)
        } else {
            Self::new(config)
        }
    }

    pub fn load_from_path(path: &Path) -> Result<Self, BrainError> {
        let bytes = fs::read(path)?;
        let is_json = path.extension().and_then(|ext| ext.to_str()) == Some("json");
        if is_json {
            let mut state: Self = serde_json::from_slice(&bytes)?;
            match state.state_version {
                STATE_VERSION => {}
                LEGACY_STATE_VERSION_V3 => {
                    state.state_version = STATE_VERSION;
                }
                LEGACY_STATE_VERSION_V2 => {
                    state.state_version = STATE_VERSION;
                }
                LEGACY_STATE_VERSION_V1 => {
                    state.state_version = STATE_VERSION;
                }
                version => return Err(BrainError::UnsupportedStateVersion(version)),
            }
            state.config.validate()?;
            state.tokenizer.rebuild_trie();
            state.rebuild_inverted_index();
            Ok(state)
        } else {
            match bincode::serde::decode_from_slice::<Self, _>(&bytes, bincode::config::standard())
            {
                Ok((mut state, _bytes_read)) => {
                    match state.state_version {
                        STATE_VERSION => {}
                        LEGACY_STATE_VERSION_V3 => {
                            state.state_version = STATE_VERSION;
                        }
                        LEGACY_STATE_VERSION_V2 => {
                            state.state_version = STATE_VERSION;
                        }
                        LEGACY_STATE_VERSION_V1 => {
                            state.state_version = STATE_VERSION;
                        }
                        version => return Err(BrainError::UnsupportedStateVersion(version)),
                    }
                    state.config.validate()?;
                    // Rebuild the trie from deserialized entries!
                    state.tokenizer.rebuild_trie();
                    state.rebuild_inverted_index();
                    Ok(state)
                }
                Err(current_error) => {
                    if let Ok((legacy, _bytes_read)) =
                        bincode::serde::decode_from_slice::<LegacyBrainStateV3, _>(
                            &bytes,
                            bincode::config::standard(),
                        )
                    {
                        if legacy.state_version != LEGACY_STATE_VERSION_V3 {
                            return Err(BrainError::UnsupportedStateVersion(legacy.state_version));
                        }
                        let mut state: Self = legacy.into();
                        state.config.validate()?;
                        state.tokenizer.rebuild_trie();
                        state.rebuild_inverted_index();
                        return Ok(state);
                    }

                    // Try Legacy V2
                    match bincode::serde::decode_from_slice::<LegacyBrainStateV2, _>(
                        &bytes,
                        bincode::config::standard(),
                    ) {
                        Ok((legacy, _bytes_read)) => {
                            if legacy.state_version != LEGACY_STATE_VERSION_V2 {
                                return Err(BrainError::UnsupportedStateVersion(
                                    legacy.state_version,
                                ));
                            }
                            let mut state: Self = legacy.into();
                            state.config.validate()?;
                            state.tokenizer.rebuild_trie();
                            state.rebuild_inverted_index();
                            Ok(state)
                        }
                        Err(_) => {
                            // Try Legacy V1
                            match bincode::serde::decode_from_slice::<LegacyBrainStateV1, _>(
                                &bytes,
                                bincode::config::standard(),
                            ) {
                                Ok((legacy, _bytes_read)) => {
                                    if legacy.state_version != LEGACY_STATE_VERSION_V1 {
                                        return Err(BrainError::UnsupportedStateVersion(
                                            legacy.state_version,
                                        ));
                                    }
                                    let mut state: Self = legacy.into();
                                    state.config.validate()?;
                                    state.tokenizer.rebuild_trie();
                                    state.rebuild_inverted_index();
                                    Ok(state)
                                }
                                Err(_) => Err(BrainError::Decode(current_error)),
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn save_to_path(&self, path: &Path) -> Result<(), BrainError> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }
        let is_json = path.extension().and_then(|ext| ext.to_str()) == Some("json");
        let payload = if is_json {
            serde_json::to_vec_pretty(self)?
        } else {
            bincode::serde::encode_to_vec(self, bincode::config::standard())?
        };
        let temp_path = temporary_state_path(path);
        fs::write(&temp_path, payload)?;
        fs::rename(&temp_path, path)?;
        Ok(())
    }
}
