use std::error::Error;
use std::fmt;

use crate::cortical::{ColumnStepError, CorticalColumn};
use crate::memory::{Hippocampus, MemoryConsolidator};
use crate::neuromodulator::Neuromodulator;
use crate::tokenizer::{SpikingTokenizer, TokenizerError};

const TOP_CONNECTIONS_LIMIT: usize = 3;

#[derive(Clone, Debug)]
pub struct SimulationConfig {
    pub input_text: String,
    pub steps_per_char: usize,
    pub context_neurons: usize,
    pub hippocampus_capacity: usize,
    pub replay_epochs: usize,
    pub lr_ltp: f32,
    pub lr_ltd: f32,
    pub neuromodulator_decay: f32,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            input_text: "spiking brain".to_string(),
            steps_per_char: 8,
            context_neurons: 8,
            hippocampus_capacity: 500,
            replay_epochs: 10,
            lr_ltp: 0.05,
            lr_ltd: 0.02,
            neuromodulator_decay: 0.85,
        }
    }
}

impl SimulationConfig {
    pub fn validate(&self) -> Result<(), SimulationError> {
        if self.input_text.trim().is_empty() {
            return Err(SimulationError::EmptyInput);
        }

        if self.steps_per_char == 0 {
            return Err(SimulationError::ZeroStepsPerChar);
        }

        if self.context_neurons == 0 {
            return Err(SimulationError::ZeroContextNeurons);
        }

        if self.hippocampus_capacity == 0 {
            return Err(SimulationError::ZeroHippocampusCapacity);
        }

        if self.replay_epochs == 0 {
            return Err(SimulationError::ZeroReplayEpochs);
        }

        validate_rate("lr_ltp", self.lr_ltp)?;
        validate_rate("lr_ltd", self.lr_ltd)?;

        if !self.neuromodulator_decay.is_finite()
            || !(0.0..=1.0).contains(&self.neuromodulator_decay)
        {
            return Err(SimulationError::InvalidDecayRate(self.neuromodulator_decay));
        }

        Ok(())
    }
}

fn validate_rate(field: &'static str, value: f32) -> Result<(), SimulationError> {
    if !value.is_finite() || value < 0.0 {
        return Err(SimulationError::InvalidLearningRate { field, value });
    }

    Ok(())
}

#[derive(Debug)]
pub enum SimulationError {
    EmptyInput,
    ZeroStepsPerChar,
    ZeroContextNeurons,
    ZeroHippocampusCapacity,
    ZeroReplayEpochs,
    InvalidLearningRate { field: &'static str, value: f32 },
    InvalidDecayRate(f32),
    Tokenizer(TokenizerError),
    ColumnStep(ColumnStepError),
}

impl fmt::Display for SimulationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "input text tidak boleh kosong"),
            Self::ZeroStepsPerChar => write!(f, "steps-per-char harus lebih besar dari 0"),
            Self::ZeroContextNeurons => write!(f, "context-neurons harus lebih besar dari 0"),
            Self::ZeroHippocampusCapacity => {
                write!(f, "hippocampus-capacity harus lebih besar dari 0")
            }
            Self::ZeroReplayEpochs => write!(f, "replay-epochs harus lebih besar dari 0"),
            Self::InvalidLearningRate { field, value } => {
                write!(f, "{field} harus bernilai finite dan >= 0, dapat: {value}")
            }
            Self::InvalidDecayRate(value) => {
                write!(
                    f,
                    "neuromodulator decay harus di antara 0.0 dan 1.0, dapat: {value}"
                )
            }
            Self::Tokenizer(err) => write!(f, "{err}"),
            Self::ColumnStep(err) => write!(f, "{err}"),
        }
    }
}

impl Error for SimulationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Tokenizer(err) => Some(err),
            Self::ColumnStep(err) => Some(err),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct NeuronSnapshot {
    pub membrane_potential: f32,
    pub threshold: f32,
    pub has_spiked: bool,
    pub refractory_steps_left: usize,
}

#[derive(Clone, Debug)]
pub struct ActiveTickSnapshot {
    pub tick: usize,
    pub total_ticks: usize,
    pub current_char: char,
    pub input_spikes: Vec<bool>,
    pub l4_spikes: Vec<bool>,
    pub l23_neurons: Vec<NeuronSnapshot>,
    pub prediction_error: f32,
    pub neuromodulator: Neuromodulator,
    pub hippocampus_size: usize,
}

#[derive(Clone, Debug)]
pub struct ConnectionSummary {
    pub source_label: String,
    pub target_label: String,
    pub weight: f32,
}

#[derive(Clone, Debug)]
pub struct SimulationReport {
    pub input_text: String,
    pub total_ticks: usize,
    pub recorded_patterns: usize,
    pub replay_ticks: usize,
    pub top_feedforward: Vec<ConnectionSummary>,
    pub top_feedback: Vec<ConnectionSummary>,
}

pub fn run_simulation(config: &SimulationConfig) -> Result<SimulationReport, SimulationError> {
    run_simulation_with_observer(config, |_| {})
}

pub fn run_simulation_with_observer<F>(
    config: &SimulationConfig,
    mut observer: F,
) -> Result<SimulationReport, SimulationError>
where
    F: FnMut(&ActiveTickSnapshot),
{
    config.validate()?;

    let tokenizer = SpikingTokenizer::new();
    let mut column = CorticalColumn::new(tokenizer.vocab_size, config.context_neurons);
    let mut neuromodulator = Neuromodulator::new(config.neuromodulator_decay);
    let mut hippocampus = Hippocampus::new(config.hippocampus_capacity);
    let consolidator = MemoryConsolidator::new();
    let input_chars: Vec<char> = config.input_text.chars().collect();
    let spike_matrix = tokenizer
        .encode(&config.input_text, config.steps_per_char)
        .map_err(SimulationError::Tokenizer)?;

    for (tick, input_spikes) in spike_matrix.iter().enumerate() {
        let current_char = input_chars[tick / config.steps_per_char];

        column
            .step(
                input_spikes,
                tick,
                &mut neuromodulator,
                config.lr_ltp,
                config.lr_ltd,
            )
            .map_err(SimulationError::ColumnStep)?;

        let l23_spikes: Vec<bool> = column
            .l23_neurons
            .iter()
            .map(|neuron| neuron.has_spiked)
            .collect();
        if l23_spikes.iter().any(|spike| *spike) {
            hippocampus.record(l23_spikes);
        }

        neuromodulator.step();
        observer(&build_tick_snapshot(
            &column,
            input_spikes,
            tick,
            spike_matrix.len(),
            current_char,
            &neuromodulator,
            hippocampus.episodic_buffer.len(),
        ));
    }

    let recorded_patterns = hippocampus.episodic_buffer.len();
    let replay_ticks = consolidator.consolidate(
        &mut column,
        &mut neuromodulator,
        &mut hippocampus,
        config.replay_epochs,
    );

    Ok(SimulationReport {
        input_text: config.input_text.clone(),
        total_ticks: spike_matrix.len(),
        recorded_patterns,
        replay_ticks,
        top_feedforward: summarize_feedforward(&column, &tokenizer, TOP_CONNECTIONS_LIMIT),
        top_feedback: summarize_feedback(&column, &tokenizer, TOP_CONNECTIONS_LIMIT),
    })
}

fn build_tick_snapshot(
    column: &CorticalColumn,
    input_spikes: &[bool],
    tick: usize,
    total_ticks: usize,
    current_char: char,
    neuromodulator: &Neuromodulator,
    hippocampus_size: usize,
) -> ActiveTickSnapshot {
    ActiveTickSnapshot {
        tick,
        total_ticks,
        current_char,
        input_spikes: input_spikes.to_vec(),
        l4_spikes: column
            .l4_neurons
            .iter()
            .map(|neuron| neuron.has_spiked)
            .collect(),
        l23_neurons: column
            .l23_neurons
            .iter()
            .map(|neuron| NeuronSnapshot {
                membrane_potential: neuron.v,
                threshold: neuron.v_threshold,
                has_spiked: neuron.has_spiked,
                refractory_steps_left: neuron.refractory_steps_left,
            })
            .collect(),
        prediction_error: column.current_prediction_error,
        neuromodulator: neuromodulator.clone(),
        hippocampus_size,
    }
}

fn summarize_feedforward(
    column: &CorticalColumn,
    tokenizer: &SpikingTokenizer,
    limit: usize,
) -> Vec<ConnectionSummary> {
    let mut synapses = column.feedforward_synapses.clone();
    synapses.sort_by(|left, right| right.weight.total_cmp(&left.weight));

    synapses
        .into_iter()
        .take(limit)
        .map(|synapse| ConnectionSummary {
            source_label: format!(
                "Input '{}' (L4 #{:02})",
                display_char(tokenizer.decode_char(synapse.pre_idx)),
                synapse.pre_idx
            ),
            target_label: format!("Context L2/3 #{:02}", synapse.post_idx),
            weight: synapse.weight,
        })
        .collect()
}

fn summarize_feedback(
    column: &CorticalColumn,
    tokenizer: &SpikingTokenizer,
    limit: usize,
) -> Vec<ConnectionSummary> {
    let mut synapses = column.feedback_synapses.clone();
    synapses.sort_by(|left, right| right.weight.total_cmp(&left.weight));

    synapses
        .into_iter()
        .take(limit)
        .map(|synapse| ConnectionSummary {
            source_label: format!("Context L2/3 #{:02}", synapse.pre_idx),
            target_label: format!(
                "Prediksi '{}' (L4 #{:02})",
                display_char(tokenizer.decode_char(synapse.post_idx)),
                synapse.post_idx
            ),
            weight: synapse.weight,
        })
        .collect()
}

fn display_char(character: char) -> char {
    if character == ' ' { '_' } else { character }
}

#[cfg(test)]
mod tests {
    use super::{SimulationConfig, SimulationError, run_simulation, run_simulation_with_observer};

    #[test]
    fn simulation_returns_summary_report() {
        let report = run_simulation(&SimulationConfig::default()).expect("simulation should pass");

        assert!(report.total_ticks > 0);
        assert!(report.recorded_patterns <= report.total_ticks);
        assert!(report.replay_ticks > 0);
        assert!(!report.top_feedforward.is_empty());
        assert!(!report.top_feedback.is_empty());
    }

    #[test]
    fn observer_is_called_once_per_active_tick() {
        let mut observed_ticks = 0usize;
        let report = run_simulation_with_observer(&SimulationConfig::default(), |_| {
            observed_ticks += 1;
        })
        .expect("simulation should pass");

        assert_eq!(observed_ticks, report.total_ticks);
    }

    #[test]
    fn simulation_rejects_blank_input() {
        let config = SimulationConfig {
            input_text: "   ".to_string(),
            ..SimulationConfig::default()
        };

        let error = run_simulation(&config).expect_err("blank input must be rejected");
        assert!(matches!(error, SimulationError::EmptyInput));
    }
}
