use std::env;
use std::fs;
use std::io::{self, BufRead, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::thread::sleep;
use std::time::{Duration, Instant};

use indicatif::{HumanDuration, ProgressBar, ProgressStyle};
use rayon::prelude::*;
use rekayasa_nural_brain::{
    ActiveTickSnapshot, BrainConfig, BrainEdgeSummary, BrainState, BrainSummary, ConnectionSummary,
    InteractionReport, SimulationConfig, SimulationError, SimulationReport, SpikingTokenizer,
    TrainingExampleReport, run_simulation, run_simulation_with_observer,
};
use serde::Deserialize;

enum Command {
    Help,
    Version,
    Simulate(SimulationRuntimeOptions),
    Chat(ChatRuntimeOptions),
    Train(TrainRuntimeOptions),
    Inspect(InspectRuntimeOptions),
    Dump(DumpRuntimeOptions),
}

struct SimulationRuntimeOptions {
    config: SimulationConfig,
    interactive: bool,
    ansi: bool,
    tick_delay_ms: u64,
}

struct ChatRuntimeOptions {
    state_path: PathBuf,
    prompt: Option<String>,
    config: BrainConfig,
    learn: bool,
}

struct TrainRuntimeOptions {
    state_path: PathBuf,
    file_path: PathBuf,
    limit: Option<usize>,
    verbose: bool,
    log_every: usize,
    show_progress: bool,
    config: BrainConfig,
}

struct InspectRuntimeOptions {
    state_path: PathBuf,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DumpMode {
    Summary,
    Tokens,
    Edges,
    Contexts,
    Memory,
    Distribution,
}

struct DumpRuntimeOptions {
    state_path: PathBuf,
    mode: DumpMode,
    prompt: Option<String>,
}

struct TrainingBatchSummary {
    examples: usize,
    prompt_tokens: usize,
    response_tokens: usize,
    max_prompt_tokens: usize,
    max_response_tokens: usize,
    new_sensor_tokens: usize,
    new_word_tokens: usize,
    new_phrase_tokens: usize,
    new_context_nodes: usize,
    new_edges: usize,
    pruned_edges: usize,
    pruned_context_nodes: usize,
}

struct TrainStateSnapshot {
    interactions: u64,
    learning_steps: u64,
    training_examples: u64,
    token_count: usize,
    sensor_token_count: usize,
    word_token_count: usize,
    phrase_token_count: usize,
    context_node_count: usize,
    edge_count: usize,
    remembered_utterances: usize,
    remembered_prompts: usize,
    remembered_pairs: usize,
}

struct TrainReporter {
    progress_bar: Option<ProgressBar>,
    total_examples: usize,
    verbose: bool,
    log_every: usize,
    started_at: Instant,
}

struct TrainingRunReport<'a> {
    original_example_count: usize,
    effective_example_count: usize,
    elapsed: Duration,
    initial_state: &'a TrainStateSnapshot,
    final_state: &'a TrainStateSnapshot,
}

type TrainingExampleTuple = (usize, String, String);

#[derive(Debug, Deserialize)]
struct PersonaChatDataset {
    train: Vec<PersonaChatDialogue>,
}

#[derive(Debug, Deserialize)]
struct PersonaChatDialogue {
    utterances: Vec<PersonaChatUtterance>,
}

#[derive(Debug, Deserialize)]
struct PersonaChatUtterance {
    candidates: Vec<String>,
    history: Vec<String>,
}

fn main() -> ExitCode {
    let command = match parse_command(env::args().skip(1)) {
        Ok(command) => command,
        Err(error) => {
            eprintln!("Error: {error}\n");
            eprintln!("{}", help_text());
            return ExitCode::FAILURE;
        }
    };

    match command {
        Command::Help => {
            println!("{}", help_text());
            ExitCode::SUCCESS
        }
        Command::Version => {
            println!("{}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        Command::Simulate(options) => run_simulate(options),
        Command::Chat(options) => run_chat(options),
        Command::Train(options) => run_train(options),
        Command::Inspect(options) => run_inspect(options),
        Command::Dump(options) => run_dump(options),
    }
}

fn parse_command<I>(args: I) -> Result<Command, String>
where
    I: IntoIterator<Item = String>,
{
    let mut args = args.into_iter();
    let Some(command) = args.next() else {
        return Ok(Command::Help);
    };

    match command.as_str() {
        "--help" | "-h" | "help" => Ok(Command::Help),
        "--version" | "-V" | "version" => Ok(Command::Version),
        "simulate" => parse_simulation_options(args).map(Command::Simulate),
        "chat" => parse_chat_options(args).map(Command::Chat),
        "train" => parse_train_options(args).map(Command::Train),
        "inspect" => parse_inspect_options(args).map(Command::Inspect),
        "dump" => parse_dump_options(args).map(Command::Dump),
        other => Err(format!("subcommand tidak dikenali: {other}")),
    }
}

fn parse_simulation_options<I>(args: I) -> Result<SimulationRuntimeOptions, String>
where
    I: IntoIterator<Item = String>,
{
    let mut args = args.into_iter();
    let mut options = SimulationRuntimeOptions {
        config: SimulationConfig::default(),
        interactive: false,
        ansi: io::stdout().is_terminal(),
        tick_delay_ms: 120,
    };

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => return Err(simulate_help_text().to_string()),
            "--input" => {
                options.config.input_text = next_value(&mut args, "--input")?;
            }
            "--steps-per-char" => {
                options.config.steps_per_char = parse_usize(
                    &next_value(&mut args, "--steps-per-char")?,
                    "--steps-per-char",
                )?;
            }
            "--context-neurons" => {
                options.config.context_neurons = parse_usize(
                    &next_value(&mut args, "--context-neurons")?,
                    "--context-neurons",
                )?;
            }
            "--hippocampus-capacity" => {
                options.config.hippocampus_capacity = parse_usize(
                    &next_value(&mut args, "--hippocampus-capacity")?,
                    "--hippocampus-capacity",
                )?;
            }
            "--replay-epochs" => {
                options.config.replay_epochs = parse_usize(
                    &next_value(&mut args, "--replay-epochs")?,
                    "--replay-epochs",
                )?;
            }
            "--lr-ltp" => {
                options.config.lr_ltp = parse_f32(&next_value(&mut args, "--lr-ltp")?, "--lr-ltp")?;
            }
            "--lr-ltd" => {
                options.config.lr_ltd = parse_f32(&next_value(&mut args, "--lr-ltd")?, "--lr-ltd")?;
            }
            "--decay" => {
                options.config.neuromodulator_decay =
                    parse_f32(&next_value(&mut args, "--decay")?, "--decay")?;
            }
            "--tick-ms" => {
                options.tick_delay_ms =
                    parse_u64(&next_value(&mut args, "--tick-ms")?, "--tick-ms")?;
            }
            "--interactive" => options.interactive = true,
            "--no-ansi" => options.ansi = false,
            value => return Err(format!("argumen simulate tidak dikenali: {value}")),
        }
    }

    if !io::stdout().is_terminal() {
        options.ansi = false;
    }

    Ok(options)
}

fn parse_chat_options<I>(args: I) -> Result<ChatRuntimeOptions, String>
where
    I: IntoIterator<Item = String>,
{
    let mut args = args.into_iter();
    let mut options = ChatRuntimeOptions {
        state_path: PathBuf::from(".brain/brain_state.bin"),
        prompt: None,
        config: BrainConfig::default(),
        learn: true,
    };

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => return Err(chat_help_text().to_string()),
            "--state" => {
                options.state_path = PathBuf::from(next_value(&mut args, "--state")?);
            }
            "--prompt" => {
                options.prompt = Some(next_value(&mut args, "--prompt")?);
            }
            "--no-learn" => {
                options.learn = false;
            }
            _ => {
                if !apply_brain_config_arg(&mut options.config, &arg, &mut args)? {
                    return Err(format!("argumen chat tidak dikenali: {arg}"));
                }
            }
        }
    }

    Ok(options)
}

fn parse_train_options<I>(args: I) -> Result<TrainRuntimeOptions, String>
where
    I: IntoIterator<Item = String>,
{
    let mut args = args.into_iter();
    let mut options = TrainRuntimeOptions {
        state_path: PathBuf::from(".brain/brain_state.bin"),
        file_path: PathBuf::from("training/id_personachat/id_personachat.json"),
        limit: None,
        verbose: false,
        log_every: 100,
        show_progress: io::stderr().is_terminal(),
        config: BrainConfig::default(),
    };

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => return Err(train_help_text().to_string()),
            "--state" => {
                options.state_path = PathBuf::from(next_value(&mut args, "--state")?);
            }
            "--file" => {
                options.file_path = PathBuf::from(next_value(&mut args, "--file")?);
            }
            "--limit" => {
                options.limit = Some(parse_non_zero_usize(
                    &next_value(&mut args, "--limit")?,
                    "--limit",
                )?);
            }
            "--verbose" => options.verbose = true,
            "--log-every" => {
                options.log_every =
                    parse_non_zero_usize(&next_value(&mut args, "--log-every")?, "--log-every")?;
            }
            "--no-progress" => options.show_progress = false,
            _ => {
                if !apply_brain_config_arg(&mut options.config, &arg, &mut args)? {
                    return Err(format!("argumen train tidak dikenali: {arg}"));
                }
            }
        }
    }

    Ok(options)
}

fn parse_inspect_options<I>(args: I) -> Result<InspectRuntimeOptions, String>
where
    I: IntoIterator<Item = String>,
{
    let mut args = args.into_iter();
    let mut options = InspectRuntimeOptions {
        state_path: PathBuf::from(".brain/brain_state.bin"),
    };

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => return Err(inspect_help_text().to_string()),
            "--state" => {
                options.state_path = PathBuf::from(next_value(&mut args, "--state")?);
            }
            value => return Err(format!("argumen inspect tidak dikenali: {value}")),
        }
    }

    Ok(options)
}

fn parse_dump_options<I>(args: I) -> Result<DumpRuntimeOptions, String>
where
    I: IntoIterator<Item = String>,
{
    let mut args = args.into_iter();
    let mut options = DumpRuntimeOptions {
        state_path: PathBuf::from(".brain/brain_state.bin"),
        mode: DumpMode::Summary,
        prompt: None,
    };

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => return Err(dump_help_text().to_string()),
            "--state" => {
                options.state_path = PathBuf::from(next_value(&mut args, "--state")?);
            }
            "--summary" => {
                options.mode = DumpMode::Summary;
            }
            "--tokens" => {
                options.mode = DumpMode::Tokens;
            }
            "--edges" => {
                options.mode = DumpMode::Edges;
            }
            "--contexts" => {
                options.mode = DumpMode::Contexts;
            }
            "--memory" => {
                options.mode = DumpMode::Memory;
            }
            "--distribution" => {
                options.mode = DumpMode::Distribution;
            }
            "--prompt" => {
                options.prompt = Some(next_value(&mut args, "--prompt")?);
            }
            value => return Err(format!("argumen dump tidak dikenali: {value}")),
        }
    }

    Ok(options)
}

fn apply_brain_config_arg<I>(
    config: &mut BrainConfig,
    arg: &str,
    args: &mut I,
) -> Result<bool, String>
where
    I: Iterator<Item = String>,
{
    match arg {
        "--word-threshold" => {
            config.word_promotion_threshold =
                parse_u64(&next_value(args, "--word-threshold")?, "--word-threshold")?;
            Ok(true)
        }
        "--phrase-threshold" => {
            config.phrase_promotion_threshold = parse_u64(
                &next_value(args, "--phrase-threshold")?,
                "--phrase-threshold",
            )?;
            Ok(true)
        }
        "--context-threshold" => {
            config.context_promotion_threshold = parse_u64(
                &next_value(args, "--context-threshold")?,
                "--context-threshold",
            )?;
            Ok(true)
        }
        "--max-ngram" => {
            config.max_ngram = parse_usize(&next_value(args, "--max-ngram")?, "--max-ngram")?;
            Ok(true)
        }
        "--context-window" => {
            config.max_context_window =
                parse_usize(&next_value(args, "--context-window")?, "--context-window")?;
            Ok(true)
        }
        "--prune-interval" => {
            config.prune_interval =
                parse_u64(&next_value(args, "--prune-interval")?, "--prune-interval")?;
            Ok(true)
        }
        "--edge-decay" => {
            config.edge_decay = parse_f32(&next_value(args, "--edge-decay")?, "--edge-decay")?;
            Ok(true)
        }
        "--min-edge-strength" => {
            config.min_edge_strength = parse_f32(
                &next_value(args, "--min-edge-strength")?,
                "--min-edge-strength",
            )?;
            Ok(true)
        }
        "--response-token-limit" => {
            config.response_token_limit = parse_usize(
                &next_value(args, "--response-token-limit")?,
                "--response-token-limit",
            )?;
            Ok(true)
        }
        "--memory-window" => {
            config.max_recent_utterances =
                parse_usize(&next_value(args, "--memory-window")?, "--memory-window")?;
            Ok(true)
        }
        "--temperature" => {
            config.generation_config.temperature = parse_f32(&next_value(args, "--temperature")?, "--temperature")?;
            Ok(true)
        }
        "--top-k" => {
            config.generation_config.top_k = parse_usize(&next_value(args, "--top-k")?, "--top-k")?;
            Ok(true)
        }
        "--top-p" => {
            config.generation_config.top_p = parse_f32(&next_value(args, "--top-p")?, "--top-p")?;
            Ok(true)
        }
        "--repetition-penalty" => {
            config.generation_config.repetition_penalty = parse_f32(&next_value(args, "--repetition-penalty")?, "--repetition-penalty")?;
            Ok(true)
        }
        "--min-confidence" => {
            config.generation_config.min_confidence = parse_f32(&next_value(args, "--min-confidence")?, "--min-confidence")?;
            Ok(true)
        }
        "--seed" => {
            config.generation_config.randomness_seed = Some(parse_u64(&next_value(args, "--seed")?, "--seed")?);
            Ok(true)
        }
        _ => Ok(false),
    }
}

fn run_simulate(options: SimulationRuntimeOptions) -> ExitCode {
    let tokenizer = SpikingTokenizer::new();
    let result = if options.interactive {
        run_simulation_with_observer(&options.config, |snapshot| {
            render_dashboard(snapshot, &tokenizer, options.ansi);
            if options.tick_delay_ms > 0 {
                sleep(Duration::from_millis(options.tick_delay_ms));
            }
        })
    } else {
        run_simulation(&options.config)
    };

    match result {
        Ok(report) => {
            print_simulation_summary(&report);
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("Simulasi gagal: {error}");
            print_simulation_hint(&error);
            ExitCode::FAILURE
        }
    }
}

fn run_chat(options: ChatRuntimeOptions) -> ExitCode {
    let mut brain = match BrainState::load_or_new(&options.state_path, options.config) {
        Ok(brain) => brain,
        Err(error) => {
            eprintln!("Gagal memuat brain state: {error}");
            return ExitCode::FAILURE;
        }
    };

    if let Some(prompt) = options.prompt {
        if options.learn {
            match brain.interact(&prompt) {
                Ok(report) => {
                    print_interaction_report(&report);
                    if let Err(error) = brain.save_to_path(&options.state_path) {
                        eprintln!("Gagal menyimpan brain state: {error}");
                        return ExitCode::FAILURE;
                    }
                    return ExitCode::SUCCESS;
                }
                Err(error) => {
                    eprintln!("Interaksi gagal: {error}");
                    return ExitCode::FAILURE;
                }
            }
        } else {
            let mut temp_brain = brain.clone();
            temp_brain.recent_utterances.clear();
            match temp_brain.generate_response(&prompt) {
                Ok(response) => {
                    println!("brain> {response}");
                    return ExitCode::SUCCESS;
                }
                Err(error) => {
                    eprintln!("Generasi respons gagal: {error}");
                    return ExitCode::FAILURE;
                }
            }
        }
    }

    println!("Dynamic persistent brain chat");
    println!("state file : {}", options.state_path.display());
    println!("perintah   : /stats, /save, /quit");
    println!();

    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let mut buffer = String::new();

    loop {
        print!("you> ");
        if io::stdout().flush().is_err() {
            eprintln!("Gagal flush stdout");
            return ExitCode::FAILURE;
        }

        buffer.clear();
        match reader.read_line(&mut buffer) {
            Ok(0) => break,
            Ok(_) => {}
            Err(error) => {
                eprintln!("Gagal membaca input: {error}");
                return ExitCode::FAILURE;
            }
        }

        let input = buffer.trim();
        if input.is_empty() {
            continue;
        }
        match input {
            "/quit" | "/exit" => break,
            "/save" => {
                if let Err(error) = brain.save_to_path(&options.state_path) {
                    eprintln!("Gagal menyimpan brain state: {error}");
                    return ExitCode::FAILURE;
                }
                println!("brain> state disimpan");
            }
            "/stats" => {
                print_brain_summary(&brain.summary(8));
            }
            _ => {
                if options.learn {
                    match brain.interact(input) {
                        Ok(report) => {
                            print_interaction_report(&report);
                            if let Err(error) = brain.save_to_path(&options.state_path) {
                                eprintln!("Gagal menyimpan brain state: {error}");
                                return ExitCode::FAILURE;
                            }
                        }
                        Err(error) => {
                            eprintln!("Interaksi gagal: {error}");
                        }
                    }
                } else {
                    let mut temp_brain = brain.clone();
                    temp_brain.recent_utterances.clear();
                    match temp_brain.generate_response(input) {
                        Ok(response) => println!("brain> {response}"),
                        Err(error) => eprintln!("Generasi respons gagal: {error}"),
                    }
                }
            },
        }
    }

    if let Err(error) = brain.save_to_path(&options.state_path) {
        eprintln!("Gagal menyimpan brain state: {error}");
        return ExitCode::FAILURE;
    }

    ExitCode::SUCCESS
}

fn run_train(options: TrainRuntimeOptions) -> ExitCode {
    let mut brain = match BrainState::load_or_new(&options.state_path, options.config) {
        Ok(brain) => brain,
        Err(error) => {
            eprintln!("Gagal memuat brain state: {error}");
            return ExitCode::FAILURE;
        }
    };

    let initial_state = capture_train_state(&brain);
    let mut examples = match load_training_examples(&options.file_path) {
        Ok(examples) => examples,
        Err(error) => {
            eprintln!(
                "Gagal membaca training file {}: {error}",
                options.file_path.display()
            );
            return ExitCode::FAILURE;
        }
    };

    if examples.is_empty() {
        eprintln!(
            "Training file {} tidak berisi pasangan prompt-response.",
            options.file_path.display()
        );
        return ExitCode::FAILURE;
    }

    let original_example_count = examples.len();
    if let Some(limit) = options.limit
        && examples.len() > limit
    {
        examples.truncate(limit);
    }
    let effective_example_count = examples.len();

    let reporter = TrainReporter::new(
        effective_example_count,
        options.verbose,
        options.log_every,
        options.show_progress,
    );
    print_training_start(
        &reporter,
        &options.file_path,
        &options.state_path,
        original_example_count,
        effective_example_count,
        &initial_state,
    );

    let mut summary = TrainingBatchSummary {
        examples: 0,
        prompt_tokens: 0,
        response_tokens: 0,
        max_prompt_tokens: 0,
        max_response_tokens: 0,
        new_sensor_tokens: 0,
        new_word_tokens: 0,
        new_phrase_tokens: 0,
        new_context_nodes: 0,
        new_edges: 0,
        pruned_edges: 0,
        pruned_context_nodes: 0,
    };

    for (index, (line_number, prompt, response)) in examples.into_iter().enumerate() {
        let example_index = index + 1;
        let report = match brain.train_pair(&prompt, &response) {
            Ok(report) => report,
            Err(error) => {
                reporter.abort();
                eprintln!("Training gagal pada baris {line_number}: {error}");
                return ExitCode::FAILURE;
            }
        };
        accumulate_training_summary(&mut summary, &report);
        reporter.record(example_index, line_number, &report, &summary, &brain);

        if reporter.should_checkpoint(example_index) {
            if let Err(error) = brain.save_to_path(&options.state_path) {
                reporter.abort();
                eprintln!("Gagal menyimpan checkpoint brain state: {error}");
                return ExitCode::FAILURE;
            }
            reporter.record_checkpoint_save(example_index, &options.state_path, &brain);
        }
    }

    reporter.finish();
    let final_state = capture_train_state(&brain);
    let run_report = TrainingRunReport {
        original_example_count,
        effective_example_count,
        elapsed: reporter.elapsed(),
        initial_state: &initial_state,
        final_state: &final_state,
    };
    print_training_summary(
        &summary,
        &options.file_path,
        &options.state_path,
        &run_report,
    );
    ExitCode::SUCCESS
}

fn run_inspect(options: InspectRuntimeOptions) -> ExitCode {
    let brain = match BrainState::load_from_path(&options.state_path) {
        Ok(brain) => brain,
        Err(error) => {
            eprintln!(
                "Gagal membaca brain state {}: {error}",
                options.state_path.display()
            );
            return ExitCode::FAILURE;
        }
    };

    print_brain_summary(&brain.summary(12));
    ExitCode::SUCCESS
}

fn run_dump(options: DumpRuntimeOptions) -> ExitCode {
    let brain = match BrainState::load_from_path(&options.state_path) {
        Ok(brain) => brain,
        Err(error) => {
            eprintln!(
                "Gagal membaca brain state {}: {error}",
                options.state_path.display()
            );
            return ExitCode::FAILURE;
        }
    };

    match options.mode {
        DumpMode::Summary => {
            print_brain_summary(&brain.summary(20));
        }
        DumpMode::Tokens => {
            println!("=== TOKENS DUMP (total: {}) ===", brain.tokenizer.entries.len());
            for (token_id, entry) in &brain.tokenizer.entries {
                println!(
                    "#{:<5} | {:<8?} | Occ: {:<5} | {:?}",
                    token_id, entry.level, entry.occurrence_count, entry.text
                );
            }
        }
        DumpMode::Edges => {
            println!("=== EDGES DUMP (total: {}) ===", brain.edges.len());
            for edge in brain.edges.values() {
                let src_lbl = brain.node_label(edge.source);
                let tgt_lbl = brain.node_label(edge.target);
                println!(
                    "{:<25} -> {:<25} | Kind: {:<18?} | Strength: {:.4} | Occ: {}",
                    src_lbl, tgt_lbl, edge.kind, edge.strength, edge.activation_count
                );
            }
        }
        DumpMode::Contexts => {
            println!("=== CONTEXTS DUMP (total: {}) ===", brain.context_patterns.len());
            for pattern in brain.context_patterns.values() {
                println!(
                    "Pattern: {:<35} | Occ: {:<5} | Node ID: {:?}",
                    pattern.label, pattern.occurrence_count, pattern.node_id
                );
                for (pred_tok, count) in &pattern.predicted_counts {
                    let pred_lbl = brain.node_label(*pred_tok);
                    println!("  -> predicting {:<25} | count: {}", pred_lbl, count);
                }
            }
        }
        DumpMode::Memory => {
            println!("=== MEMORY DUMP ===");
            println!("--- Prompt Response Memory (total keys: {}) ---", brain.prompt_response_memory.len());
            for (prompt, responses) in &brain.prompt_response_memory {
                println!("Prompt: {:?}", prompt);
                for (response, count) in responses {
                    println!("  -> Response: {:<40} | count: {}", response, count);
                }
            }
            println!("\n--- Recent Utterances (total: {}) ---", brain.recent_utterances.len());
            for utterance in &brain.recent_utterances {
                println!(
                    "  [{}] {:?}",
                    utterance.interaction_index, utterance.text
                );
            }
        }
        DumpMode::Distribution => {
            let prompt_text = options.prompt.clone().or_else(|| {
                brain.recent_utterances.back().map(|u| u.text.clone())
            });
            if let Some(text) = prompt_text {
                println!("=== DISTRIBUTION DUMP FOR PROMPT: {:?} ===", text);
                let normalized = rekayasa_nural_brain::brain::learning::normalize_input(&text).unwrap_or_default();
                let token_ids = brain.tokenizer.tokenize(&normalized);
                let distribution = brain.next_token_distribution(&token_ids, &[], &[], &brain.config.generation_config);
                println!("Candidates (total: {}):", distribution.len());
                for candidate in distribution {
                    let lbl = brain.node_label(candidate.token_id);
                    println!(
                        "  Token: {:<20} (ID: #{}) | Prob: {:.4} | Score: {:.4} | Source: {:?}",
                        lbl, candidate.token_id, candidate.probability, candidate.score, candidate.source
                    );
                }
            } else {
                println!("Gunakan --prompt <teks> untuk menampilkan distribusi probabilitas token.");
            }
        }
    }
    ExitCode::SUCCESS
}

fn next_value<I>(args: &mut I, flag: &str) -> Result<String, String>
where
    I: Iterator<Item = String>,
{
    args.next()
        .ok_or_else(|| format!("nilai untuk {flag} belum diberikan"))
}

fn parse_usize(value: &str, flag: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|_| format!("{flag} harus berupa integer tak negatif"))
}

fn parse_non_zero_usize(value: &str, flag: &str) -> Result<usize, String> {
    let parsed = parse_usize(value, flag)?;
    if parsed == 0 {
        Err(format!("{flag} harus lebih besar dari 0"))
    } else {
        Ok(parsed)
    }
}

fn parse_u64(value: &str, flag: &str) -> Result<u64, String> {
    value
        .parse::<u64>()
        .map_err(|_| format!("{flag} harus berupa integer tak negatif"))
}

fn parse_f32(value: &str, flag: &str) -> Result<f32, String> {
    value
        .parse::<f32>()
        .map_err(|_| format!("{flag} harus berupa angka desimal yang valid"))
}

fn render_dashboard(snapshot: &ActiveTickSnapshot, tokenizer: &SpikingTokenizer, ansi: bool) {
    if ansi {
        print!("\x1b[2J\x1b[H");
    }

    let sensory_line = snapshot
        .input_spikes
        .iter()
        .enumerate()
        .map(|(index, input_spike)| {
            let fired = snapshot.l4_spikes.get(index).copied().unwrap_or(false);
            if fired {
                display_char(tokenizer.decode_char(index)).to_ascii_uppercase()
            } else if *input_spike {
                '*'
            } else {
                '.'
            }
        })
        .collect::<String>();

    println!("PSCM ACTIVE SIMULATION");
    println!(
        "tick {:03}/{:03} | input '{}' | prediction error {:.2}",
        snapshot.tick + 1,
        snapshot.total_ticks,
        display_char(snapshot.current_char),
        snapshot.prediction_error
    );
    println!(
        "ACh {:.2} | DA {:.2} | hippocampus {}",
        snapshot.neuromodulator.acetylcholine,
        snapshot.neuromodulator.dopamine,
        snapshot.hippocampus_size
    );
    println!("L4 sensory: {sensory_line}");

    for (index, neuron) in snapshot.l23_neurons.iter().enumerate() {
        let bar_length = (neuron.membrane_potential * 10.0).clamp(0.0, 10.0) as usize;
        let bar = format!("{}{}", "#".repeat(bar_length), "-".repeat(10 - bar_length));
        let status = if neuron.has_spiked {
            "SPIKE"
        } else if neuron.refractory_steps_left > 0 {
            "REFRA"
        } else {
            "IDLE "
        };
        println!(
            "L23 #{:02} [{}] V:[{}] threshold {:.2}",
            index, status, bar, neuron.threshold
        );
    }

    println!();
    let _ = io::stdout().flush();
}

fn print_simulation_summary(report: &SimulationReport) {
    println!("PSCM simulation completed");
    println!("input text           : {}", report.input_text);
    println!("active ticks         : {}", report.total_ticks);
    println!("recorded patterns    : {}", report.recorded_patterns);
    println!("replay ticks         : {}", report.replay_ticks);
    println!();
    println!("Top feedforward connections:");
    print_connections(&report.top_feedforward);
    println!();
    println!("Top feedback connections:");
    print_connections(&report.top_feedback);
}

fn print_connections(connections: &[ConnectionSummary]) {
    for connection in connections {
        println!(
            "- {} -> {} | weight {:.2}",
            connection.source_label, connection.target_label, connection.weight
        );
    }
}

fn print_interaction_report(report: &InteractionReport) {
    println!("brain> {}", report.response);

    let learning = &report.learning;
    if !learning.new_sensor_tokens.is_empty()
        || !learning.new_word_tokens.is_empty()
        || !learning.new_phrase_tokens.is_empty()
        || !learning.new_context_nodes.is_empty()
        || learning.pruned_edges > 0
        || learning.pruned_context_nodes > 0
    {
        println!(
            "growth> sensor +{} | word +{} | phrase +{} | context +{} | edges +{} | pruned edges {} | pruned context {}",
            learning.new_sensor_tokens.len(),
            learning.new_word_tokens.len(),
            learning.new_phrase_tokens.len(),
            learning.new_context_nodes.len(),
            learning.new_edges,
            learning.pruned_edges,
            learning.pruned_context_nodes
        );
    }
}

fn accumulate_training_summary(summary: &mut TrainingBatchSummary, report: &TrainingExampleReport) {
    summary.examples += 1;
    summary.prompt_tokens += report.prompt_token_count;
    summary.response_tokens += report.response_token_count;
    summary.max_prompt_tokens = summary.max_prompt_tokens.max(report.prompt_token_count);
    summary.max_response_tokens = summary.max_response_tokens.max(report.response_token_count);
    summary.new_sensor_tokens += report.new_sensor_tokens.len();
    summary.new_word_tokens += report.new_word_tokens.len();
    summary.new_phrase_tokens += report.new_phrase_tokens.len();
    summary.new_context_nodes += report.new_context_nodes.len();
    summary.new_edges += report.new_edges;
    summary.pruned_edges += report.pruned_edges;
    summary.pruned_context_nodes += report.pruned_context_nodes;
}

fn print_training_start(
    reporter: &TrainReporter,
    file_path: &Path,
    state_path: &Path,
    original_example_count: usize,
    effective_example_count: usize,
    initial_state: &TrainStateSnapshot,
) {
    reporter.println("Training started");
    reporter.println(&format!("training file       : {}", file_path.display()));
    reporter.println(&format!("state file          : {}", state_path.display()));
    reporter.println(&format!("dataset examples    : {}", original_example_count));
    reporter.println(&format!(
        "effective examples  : {}",
        effective_example_count
    ));
    reporter.println(&format!(
        "monitoring          : progress {} | verbose {} | log every {} | checkpoint save ya",
        yes_no(reporter.progress_bar.is_some()),
        yes_no(reporter.verbose),
        reporter.log_every
    ));
    reporter.println(&format!(
        "parallel prep       : rayon workers {}",
        rayon::current_num_threads()
    ));
    reporter.println(&format!(
        "initial state       : interactions {} | tokens {} | ctx {} | edges {} | exact prompts {} | exact pairs {}",
        initial_state.interactions,
        initial_state.token_count,
        initial_state.context_node_count,
        initial_state.edge_count,
        initial_state.remembered_prompts,
        initial_state.remembered_pairs
    ));
}

fn print_training_summary(
    summary: &TrainingBatchSummary,
    file_path: &Path,
    state_path: &Path,
    run_report: &TrainingRunReport<'_>,
) {
    println!("Training completed");
    println!("training file       : {}", file_path.display());
    println!("state file          : {}", state_path.display());
    println!(
        "examples            : {} / {}",
        run_report.effective_example_count, run_report.original_example_count
    );
    println!(
        "duration            : {}",
        HumanDuration(run_report.elapsed)
    );
    println!(
        "average rate        : {:.2} examples/s",
        calculate_example_rate(summary.examples, run_report.elapsed)
    );
    println!("prompt tokens total : {}", summary.prompt_tokens);
    println!("response tokens total: {}", summary.response_tokens);
    println!(
        "avg prompt tokens   : {:.2}",
        average_tokens(summary.prompt_tokens, summary.examples)
    );
    println!(
        "avg response tokens : {:.2}",
        average_tokens(summary.response_tokens, summary.examples)
    );
    println!("max prompt tokens   : {}", summary.max_prompt_tokens);
    println!("max response tokens : {}", summary.max_response_tokens);
    println!("new sensor tokens   : {}", summary.new_sensor_tokens);
    println!("new word tokens     : {}", summary.new_word_tokens);
    println!("new phrase tokens   : {}", summary.new_phrase_tokens);
    println!("new context nodes   : {}", summary.new_context_nodes);
    println!("new edges           : {}", summary.new_edges);
    println!("pruned edges        : {}", summary.pruned_edges);
    println!("pruned context      : {}", summary.pruned_context_nodes);
    println!(
        "interactions total  : {} -> {} (+{})",
        run_report.initial_state.interactions,
        run_report.final_state.interactions,
        run_report
            .final_state
            .interactions
            .saturating_sub(run_report.initial_state.interactions)
    );
    println!(
        "learning steps      : {} -> {} (+{})",
        run_report.initial_state.learning_steps,
        run_report.final_state.learning_steps,
        run_report
            .final_state
            .learning_steps
            .saturating_sub(run_report.initial_state.learning_steps)
    );
    println!(
        "training examples   : {} -> {} (+{})",
        run_report.initial_state.training_examples,
        run_report.final_state.training_examples,
        run_report
            .final_state
            .training_examples
            .saturating_sub(run_report.initial_state.training_examples)
    );
    println!(
        "tokens total        : {} -> {} (+{})",
        run_report.initial_state.token_count,
        run_report.final_state.token_count,
        run_report
            .final_state
            .token_count
            .saturating_sub(run_report.initial_state.token_count)
    );
    println!(
        "sensor tokens       : {} -> {} (+{})",
        run_report.initial_state.sensor_token_count,
        run_report.final_state.sensor_token_count,
        run_report
            .final_state
            .sensor_token_count
            .saturating_sub(run_report.initial_state.sensor_token_count)
    );
    println!(
        "word tokens         : {} -> {} (+{})",
        run_report.initial_state.word_token_count,
        run_report.final_state.word_token_count,
        run_report
            .final_state
            .word_token_count
            .saturating_sub(run_report.initial_state.word_token_count)
    );
    println!(
        "phrase tokens       : {} -> {} (+{})",
        run_report.initial_state.phrase_token_count,
        run_report.final_state.phrase_token_count,
        run_report
            .final_state
            .phrase_token_count
            .saturating_sub(run_report.initial_state.phrase_token_count)
    );
    println!(
        "context nodes total : {} -> {} (+{})",
        run_report.initial_state.context_node_count,
        run_report.final_state.context_node_count,
        run_report
            .final_state
            .context_node_count
            .saturating_sub(run_report.initial_state.context_node_count)
    );
    println!(
        "edges total         : {} -> {} (+{})",
        run_report.initial_state.edge_count,
        run_report.final_state.edge_count,
        run_report
            .final_state
            .edge_count
            .saturating_sub(run_report.initial_state.edge_count)
    );
    println!(
        "remembered prompts  : {} -> {} (+{})",
        run_report.initial_state.remembered_prompts,
        run_report.final_state.remembered_prompts,
        run_report
            .final_state
            .remembered_prompts
            .saturating_sub(run_report.initial_state.remembered_prompts)
    );
    println!(
        "remembered pairs    : {} -> {} (+{})",
        run_report.initial_state.remembered_pairs,
        run_report.final_state.remembered_pairs,
        run_report
            .final_state
            .remembered_pairs
            .saturating_sub(run_report.initial_state.remembered_pairs)
    );
    println!(
        "remembered inputs   : {} -> {} (+{})",
        run_report.initial_state.remembered_utterances,
        run_report.final_state.remembered_utterances,
        run_report
            .final_state
            .remembered_utterances
            .saturating_sub(run_report.initial_state.remembered_utterances)
    );
}

impl TrainReporter {
    fn new(total_examples: usize, verbose: bool, log_every: usize, show_progress: bool) -> Self {
        let progress_bar = if show_progress {
            let bar = ProgressBar::new(total_examples as u64);
            bar.set_style(
                ProgressStyle::with_template(
                    "[{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos:>7}/{len:7} {percent:>3}% | {per_sec:>12} | ETA {eta_precise} | {msg}",
                )
                .expect("progress bar template harus valid")
                .progress_chars("=> "),
            );
            bar.enable_steady_tick(Duration::from_millis(120));
            Some(bar)
        } else {
            None
        };

        Self {
            progress_bar,
            total_examples,
            verbose,
            log_every,
            started_at: Instant::now(),
        }
    }

    fn record(
        &self,
        example_index: usize,
        line_number: usize,
        report: &TrainingExampleReport,
        summary: &TrainingBatchSummary,
        brain: &BrainState,
    ) {
        if let Some(progress_bar) = &self.progress_bar {
            progress_bar.inc(1);
            progress_bar.set_message(format!(
                "last +s{} +w{} +p{} +c{} +e{} -e{} -c{}",
                report.new_sensor_tokens.len(),
                report.new_word_tokens.len(),
                report.new_phrase_tokens.len(),
                report.new_context_nodes.len(),
                report.new_edges,
                report.pruned_edges,
                report.pruned_context_nodes
            ));
        }

        if !self.should_checkpoint(example_index) {
            return;
        }

        let elapsed = self.elapsed();
        let state = capture_train_state(brain);
        self.println(&format!(
            "checkpoint {:>7}/{:>7} | sumber #{line_number} | elapsed {} | rate {:.2} ex/s | prompt tok {} | response tok {} | growth +s{} +w{} +p{} +c{} +e{} -e{} -c{} | state tok {} ctx {} edge {} exact {}",
            example_index,
            self.total_examples,
            HumanDuration(elapsed),
            calculate_example_rate(example_index, elapsed),
            report.prompt_token_count,
            report.response_token_count,
            report.new_sensor_tokens.len(),
            report.new_word_tokens.len(),
            report.new_phrase_tokens.len(),
            report.new_context_nodes.len(),
            report.new_edges,
            report.pruned_edges,
            report.pruned_context_nodes,
            state.token_count,
            state.context_node_count,
            state.edge_count,
            state.remembered_pairs
        ));
        self.println(&format!(
            "cumulative          : examples {} | prompt tok {} | response tok {} | new sensor {} | new word {} | new phrase {} | new context {} | new edges {} | pruned edges {} | pruned context {}",
            summary.examples,
            summary.prompt_tokens,
            summary.response_tokens,
            summary.new_sensor_tokens,
            summary.new_word_tokens,
            summary.new_phrase_tokens,
            summary.new_context_nodes,
            summary.new_edges,
            summary.pruned_edges,
            summary.pruned_context_nodes
        ));

        if self.verbose {
            self.println(&format!(
                "prompt> {}",
                truncate_for_log(&report.prompt, 120)
            ));
            self.println(&format!(
                "response> {}",
                truncate_for_log(&report.response, 120)
            ));
        }
    }

    fn should_checkpoint(&self, example_index: usize) -> bool {
        example_index == 1
            || example_index == self.total_examples
            || example_index.is_multiple_of(self.log_every)
    }

    fn record_checkpoint_save(&self, example_index: usize, state_path: &Path, brain: &BrainState) {
        let state = capture_train_state(brain);
        self.println(&format!(
            "checkpoint save     : {:>7}/{:>7} | {} | interactions {} | tokens {} | edges {}",
            example_index,
            self.total_examples,
            state_path.display(),
            state.interactions,
            state.token_count,
            state.edge_count
        ));
    }

    fn println(&self, line: &str) {
        if let Some(progress_bar) = &self.progress_bar {
            progress_bar.println(line);
        } else {
            println!("{line}");
        }
    }

    fn finish(&self) {
        if let Some(progress_bar) = &self.progress_bar {
            progress_bar.finish_with_message("training selesai");
        }
    }

    fn abort(&self) {
        if let Some(progress_bar) = &self.progress_bar {
            progress_bar.abandon_with_message("training dibatalkan");
        }
    }

    fn elapsed(&self) -> Duration {
        self.started_at.elapsed()
    }
}

fn capture_train_state(brain: &BrainState) -> TrainStateSnapshot {
    let mut sensor_token_count = 0;
    let mut word_token_count = 0;
    let mut phrase_token_count = 0;

    for token in brain.tokenizer.entries.values() {
        match token.level {
            rekayasa_nural_brain::brain::TokenLevel::Sensor => sensor_token_count += 1,
            rekayasa_nural_brain::brain::TokenLevel::Word => word_token_count += 1,
            rekayasa_nural_brain::brain::TokenLevel::Phrase => phrase_token_count += 1,
        }
    }

    TrainStateSnapshot {
        interactions: brain.interaction_count,
        learning_steps: brain.learning_step_count,
        training_examples: brain.training_example_count,
        token_count: brain.tokenizer.entries.len(),
        sensor_token_count,
        word_token_count,
        phrase_token_count,
        context_node_count: brain
            .nodes
            .values()
            .filter(|node| node.kind == rekayasa_nural_brain::brain::NodeKind::Context)
            .count(),
        edge_count: brain.edges.len(),
        remembered_utterances: brain.recent_utterances.len(),
        remembered_prompts: brain.prompt_response_memory.len(),
        remembered_pairs: brain
            .prompt_response_memory
            .values()
            .map(|responses| responses.len())
            .sum(),
    }
}

fn average_tokens(total_tokens: usize, examples: usize) -> f64 {
    if examples == 0 {
        0.0
    } else {
        total_tokens as f64 / examples as f64
    }
}

fn calculate_example_rate(examples: usize, elapsed: Duration) -> f64 {
    let seconds = elapsed.as_secs_f64();
    if examples == 0 || seconds <= f64::EPSILON {
        0.0
    } else {
        examples as f64 / seconds
    }
}

fn truncate_for_log(text: &str, max_chars: usize) -> String {
    let mut output = String::new();
    for (index, character) in text.chars().enumerate() {
        if index >= max_chars {
            output.push_str("...");
            break;
        }
        output.push(character);
    }
    output
}

fn yes_no(value: bool) -> &'static str {
    if value { "ya" } else { "tidak" }
}

fn print_brain_summary(summary: &BrainSummary) {
    println!("Brain summary");
    println!("interactions        : {}", summary.interactions);
    println!("learning steps      : {}", summary.learning_steps);
    println!("training examples   : {}", summary.training_examples);
    println!("tokens total        : {}", summary.token_count);
    println!("sensor tokens       : {}", summary.sensor_token_count);
    println!("word tokens         : {}", summary.word_token_count);
    println!("phrase tokens       : {}", summary.phrase_token_count);
    println!("context nodes       : {}", summary.context_node_count);
    println!("edges               : {}", summary.edge_count);
    println!("remembered inputs   : {}", summary.remembered_utterances);
    if !summary.latest_tokens.is_empty() {
        println!("latest tokens       : {}", summary.latest_tokens.join(", "));
    }
    if !summary.strongest_edges.is_empty() {
        println!();
        println!("Strongest edges:");
        for edge in &summary.strongest_edges {
            print_brain_edge(edge);
        }
    }
}

fn print_brain_edge(edge: &BrainEdgeSummary) {
    println!(
        "- [{}] {} -> {} | strength {:.2} | activations {}",
        edge_kind_label(edge.kind),
        edge.source_label,
        edge.target_label,
        edge.strength,
        edge.activation_count
    );
}

fn edge_kind_label(kind: rekayasa_nural_brain::brain::EdgeKind) -> &'static str {
    match kind {
        rekayasa_nural_brain::brain::EdgeKind::Transition => "transition",
        rekayasa_nural_brain::brain::EdgeKind::ContextInput => "context-input",
        rekayasa_nural_brain::brain::EdgeKind::ContextPrediction => "context-prediction",
    }
}

fn print_simulation_hint(error: &SimulationError) {
    match error {
        SimulationError::Tokenizer(_) | SimulationError::EmptyInput => {
            eprintln!(
                "Hint: gunakan hanya karakter a-z dan spasi, misalnya simulate --input \"spiking brain\""
            );
        }
        _ => {}
    }
}

fn load_training_examples(path: &Path) -> Result<Vec<TrainingExampleTuple>, String> {
    match path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| extension.to_ascii_lowercase())
        .as_deref()
    {
        Some("json") => load_personachat_examples(path),
        _ => load_tsv_training_examples(path),
    }
}

fn load_tsv_training_examples(path: &Path) -> Result<Vec<TrainingExampleTuple>, String> {
    let content = fs::read_to_string(path).map_err(|error| error.to_string())?;
    let lines = content.lines().collect::<Vec<_>>();
    let mut examples: Vec<Result<Option<TrainingExampleTuple>, String>> = lines
        .into_par_iter()
        .enumerate()
        .map(|(index, raw_line): (usize, &str)| {
            let line_number = index + 1;
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') {
                return Ok(None);
            }

            let Some((prompt, response)) = line.split_once('\t') else {
                return Err(format!(
                    "baris {line_number} harus berbentuk 'prompt<TAB>response'"
                ));
            };

            let prompt = prompt.trim();
            let response = response.trim();
            if prompt.is_empty() || response.is_empty() {
                return Err(format!(
                    "baris {line_number} tidak boleh memiliki prompt atau response kosong"
                ));
            }

            Ok(Some((
                line_number,
                prompt.to_string(),
                response.to_string(),
            )))
        })
        .collect();

    examples.sort_by(|left, right| match (left, right) {
        (Ok(Some((left_line, _, _))), Ok(Some((right_line, _, _)))) => left_line.cmp(right_line),
        (Err(_), Ok(_)) => std::cmp::Ordering::Less,
        (Ok(_), Err(_)) => std::cmp::Ordering::Greater,
        _ => std::cmp::Ordering::Equal,
    });

    let mut ordered_examples = Vec::new();
    for example in examples {
        match example {
            Ok(Some(example)) => ordered_examples.push(example),
            Ok(None) => {}
            Err(error) => return Err(error),
        }
    }

    Ok(ordered_examples)
}

fn load_personachat_examples(path: &Path) -> Result<Vec<TrainingExampleTuple>, String> {
    let content = fs::read_to_string(path).map_err(|error| error.to_string())?;
    let dataset: PersonaChatDataset =
        serde_json::from_str(&content).map_err(|error| error.to_string())?;
    let mut examples = dataset
        .train
        .into_par_iter()
        .enumerate()
        .map(|(dialogue_index, dialogue)| {
            dialogue
                .utterances
                .into_iter()
                .enumerate()
                .filter_map(|(utterance_index, utterance)| {
                    let prompt = utterance.history.last()?.trim();
                    let response = utterance.candidates.last()?.trim();
                    if prompt.is_empty() || response.is_empty() {
                        return None;
                    }

                    Some((
                        dialogue_index,
                        utterance_index,
                        prompt.to_string(),
                        response.to_string(),
                    ))
                })
                .collect::<Vec<_>>()
        })
        .flatten()
        .collect::<Vec<_>>();

    if examples.is_empty() {
        return Err("dataset JSON tidak menghasilkan pasangan prompt-response".to_string());
    }

    examples.sort_by(
        |(left_dialogue, left_utterance, _, _), (right_dialogue, right_utterance, _, _)| {
            left_dialogue
                .cmp(right_dialogue)
                .then_with(|| left_utterance.cmp(right_utterance))
        },
    );

    Ok(examples
        .into_iter()
        .enumerate()
        .map(
            |(index, (_dialogue_index, _utterance_index, prompt, response))| {
                (index + 1, prompt, response)
            },
        )
        .collect())
}

fn help_text() -> &'static str {
    "\
rekayasa_nural_brain

Usage:
  cargo run -- <command> [options]

Commands:
  chat        Jalankan dynamic persistent brain yang belajar dari interaksi
  train       Latih brain dari file pasangan prompt-response
  inspect     Lihat ringkasan state brain yang tersimpan
  dump        Dump vocabulary, edges, atau summary dalam format terperinci
  simulate    Jalankan simulator PSCM fixed-size lama
  help        Tampilkan bantuan
  version     Tampilkan versi

Examples:
  cargo run -- train
  cargo run -- train --file training/id_personachat/id_personachat.json
  cargo run -- chat
  cargo run -- chat --prompt \"saya suka kopi\"
  cargo run -- inspect
  cargo run -- dump --summary
  cargo run -- dump --tokens
  cargo run -- dump --edges
  cargo run -- simulate --input \"spiking brain\"
"
}

fn chat_help_text() -> &'static str {
    "\
Usage:
  cargo run -- chat [options]

Options:
  --state <path>                  Lokasi file state brain
  --prompt <text>                 Jalankan satu interaksi lalu keluar
  --word-threshold <n>            Batas kemunculan sebelum kata dipromosikan
  --phrase-threshold <n>          Batas kemunculan sebelum frasa dipromosikan
  --context-threshold <n>         Batas kemunculan sebelum context node dibuat
  --max-ngram <n>                 Panjang frasa maksimum untuk promosi
  --context-window <n>            Panjang window konteks maksimum
  --prune-interval <n>            Frekuensi pruning graph
  --edge-decay <value>            Faktor decay edge saat pruning
  --min-edge-strength <value>     Ambang hapus edge lemah
  --response-token-limit <n>      Batas token balasan yang digenerasi
  --memory-window <n>             Jumlah input yang diingat

Examples:
  cargo run -- chat
  cargo run -- chat --prompt \"halo brain\"
  cargo run -- chat --state data/brain.bin --word-threshold 2 --phrase-threshold 3
"
}

fn train_help_text() -> &'static str {
    "\
Usage:
  cargo run -- train [options]

Options:
  --file <path>                   File training JSON atau TSV
  --state <path>                  Lokasi file state brain
  --limit <n>                     Batasi jumlah contoh training yang diproses
  --verbose                       Tampilkan prompt-response di checkpoint log
  --log-every <n>                 Cetak checkpoint detail dan save .bin tiap n contoh
  --no-progress                   Nonaktifkan progress bar terminal
  --word-threshold <n>            Batas kemunculan sebelum kata dipromosikan
  --phrase-threshold <n>          Batas kemunculan sebelum frasa dipromosikan
  --context-threshold <n>         Batas kemunculan sebelum context node dibuat
  --max-ngram <n>                 Panjang frasa maksimum untuk promosi
  --context-window <n>            Panjang window konteks maksimum
  --prune-interval <n>            Frekuensi pruning graph
  --edge-decay <value>            Faktor decay edge saat pruning
  --min-edge-strength <value>     Ambang hapus edge lemah
  --response-token-limit <n>      Batas token balasan yang digenerasi
  --memory-window <n>             Jumlah input yang diingat

Format file:
  JSON PersonaChat atau prompt<TAB>response

Catatan:
  Preparasi dataset menggunakan rayon secara paralel.
  Update brain tetap diproses berurutan agar state belajar tetap konsisten.

Examples:
  cargo run -- train
  cargo run -- train --file training/id_personachat/id_personachat.json
  cargo run -- train --file training/id_personachat/id_personachat.json --limit 500
  cargo run -- train --verbose --log-every 25
  cargo run -- train --state data/brain.bin --file training/id_personachat/id_personachat.json
"
}

fn inspect_help_text() -> &'static str {
    "\
Usage:
  cargo run -- inspect [options]

Options:
  --state <path>                  Lokasi file state brain
"
}

fn dump_help_text() -> &'static str {
    "\
Usage:
  cargo run -- dump [options]

Options:
  --state <path>                  Lokasi file state brain
  --summary                       Dump ringkasan state brain (default)
  --tokens                        Dump seluruh daftar token vocabulary terperinci
  --edges                         Dump seluruh edge network graph terperinci
  --contexts                      Dump seluruh context patterns beserta prediksi
  --memory                        Dump recent memory dan prompt-response memory
  --distribution                  Dump distribusi probabilitas token berikutnya
  --prompt <teks>                 Prompt kustom untuk dump --distribution
"
}

fn simulate_help_text() -> &'static str {
    "\
Usage:
  cargo run -- simulate [options]

Options:
  --input <text>                  Teks input untuk simulasi
  --steps-per-char <n>            Jumlah tick per karakter
  --context-neurons <n>           Jumlah neuron konteks L2/3
  --hippocampus-capacity <n>      Kapasitas buffer hipokampus
  --replay-epochs <n>             Jumlah epoch replay tidur
  --lr-ltp <value>                Learning rate LTP
  --lr-ltd <value>                Learning rate LTD
  --decay <value>                 Decay neuromodulator (0.0 - 1.0)
  --interactive                   Tampilkan dashboard tick-by-tick
  --tick-ms <n>                   Delay antar tick saat mode interaktif
  --no-ansi                       Nonaktifkan clear-screen ANSI
"
}

fn display_char(character: char) -> char {
    if character == ' ' { '_' } else { character }
}
