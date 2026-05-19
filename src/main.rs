use std::env;
use std::io::{self, BufRead, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::thread::sleep;
use std::time::Duration;

use rekayasa_nural_brain::{
    ActiveTickSnapshot, BrainConfig, BrainEdgeSummary, BrainState, BrainSummary, ConnectionSummary,
    InteractionReport, SimulationConfig, SimulationError, SimulationReport, SpikingTokenizer,
    run_simulation, run_simulation_with_observer,
};

enum Command {
    Help,
    Version,
    Simulate(SimulationRuntimeOptions),
    Chat(ChatRuntimeOptions),
    Inspect(InspectRuntimeOptions),
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
}

struct InspectRuntimeOptions {
    state_path: PathBuf,
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
        Command::Inspect(options) => run_inspect(options),
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
        "inspect" => parse_inspect_options(args).map(Command::Inspect),
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
            "--word-threshold" => {
                options.config.word_promotion_threshold = parse_u64(
                    &next_value(&mut args, "--word-threshold")?,
                    "--word-threshold",
                )?;
            }
            "--phrase-threshold" => {
                options.config.phrase_promotion_threshold = parse_u64(
                    &next_value(&mut args, "--phrase-threshold")?,
                    "--phrase-threshold",
                )?;
            }
            "--context-threshold" => {
                options.config.context_promotion_threshold = parse_u64(
                    &next_value(&mut args, "--context-threshold")?,
                    "--context-threshold",
                )?;
            }
            "--max-ngram" => {
                options.config.max_ngram =
                    parse_usize(&next_value(&mut args, "--max-ngram")?, "--max-ngram")?;
            }
            "--context-window" => {
                options.config.max_context_window = parse_usize(
                    &next_value(&mut args, "--context-window")?,
                    "--context-window",
                )?;
            }
            "--prune-interval" => {
                options.config.prune_interval = parse_u64(
                    &next_value(&mut args, "--prune-interval")?,
                    "--prune-interval",
                )?;
            }
            "--edge-decay" => {
                options.config.edge_decay =
                    parse_f32(&next_value(&mut args, "--edge-decay")?, "--edge-decay")?;
            }
            "--min-edge-strength" => {
                options.config.min_edge_strength = parse_f32(
                    &next_value(&mut args, "--min-edge-strength")?,
                    "--min-edge-strength",
                )?;
            }
            "--response-token-limit" => {
                options.config.response_token_limit = parse_usize(
                    &next_value(&mut args, "--response-token-limit")?,
                    "--response-token-limit",
                )?;
            }
            "--memory-window" => {
                options.config.max_recent_utterances = parse_usize(
                    &next_value(&mut args, "--memory-window")?,
                    "--memory-window",
                )?;
            }
            value => return Err(format!("argumen chat tidak dikenali: {value}")),
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
            _ => match brain.interact(input) {
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
            },
        }
    }

    if let Err(error) = brain.save_to_path(&options.state_path) {
        eprintln!("Gagal menyimpan brain state: {error}");
        return ExitCode::FAILURE;
    }

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

fn print_brain_summary(summary: &BrainSummary) {
    println!("Brain summary");
    println!("interactions        : {}", summary.interactions);
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

fn help_text() -> &'static str {
    "\
rekayasa_nural_brain

Usage:
  cargo run -- <command> [options]

Commands:
  chat        Jalankan dynamic persistent brain yang belajar dari interaksi
  inspect     Lihat ringkasan state brain yang tersimpan
  simulate    Jalankan simulator PSCM fixed-size lama
  help        Tampilkan bantuan
  version     Tampilkan versi

Examples:
  cargo run -- chat
  cargo run -- chat --prompt \"saya suka kopi\"
  cargo run -- inspect
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

fn inspect_help_text() -> &'static str {
    "\
Usage:
  cargo run -- inspect [options]

Options:
  --state <path>                  Lokasi file state brain
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

#[allow(dead_code)]
fn _state_exists(path: &Path) -> bool {
    path.exists()
}
