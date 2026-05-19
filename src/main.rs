use std::env;
use std::io::{self, IsTerminal, Write};
use std::process::ExitCode;
use std::thread::sleep;
use std::time::Duration;

use rekayasa_nural_brain::{
    ActiveTickSnapshot, ConnectionSummary, SimulationConfig, SimulationError, SimulationReport,
    SpikingTokenizer, run_simulation, run_simulation_with_observer,
};

struct RuntimeOptions {
    config: SimulationConfig,
    interactive: bool,
    ansi: bool,
    tick_delay_ms: u64,
}

fn main() -> ExitCode {
    let options = match parse_args(env::args().skip(1)) {
        Ok(Some(options)) => options,
        Ok(None) => return ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("Error: {error}\n");
            eprintln!("{}", help_text());
            return ExitCode::FAILURE;
        }
    };

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
            print_summary(&report);
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("Simulasi gagal: {error}");
            print_runtime_hint(&error);
            ExitCode::FAILURE
        }
    }
}

fn parse_args<I>(args: I) -> Result<Option<RuntimeOptions>, String>
where
    I: IntoIterator<Item = String>,
{
    let mut args = args.into_iter();
    let mut options = RuntimeOptions {
        config: SimulationConfig::default(),
        interactive: false,
        ansi: io::stdout().is_terminal(),
        tick_delay_ms: 120,
    };

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                println!("{}", help_text());
                return Ok(None);
            }
            "--version" | "-V" => {
                println!("{}", env!("CARGO_PKG_VERSION"));
                return Ok(None);
            }
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
            value => return Err(format!("argumen tidak dikenali: {value}")),
        }
    }

    if !io::stdout().is_terminal() {
        options.ansi = false;
    }

    Ok(Some(options))
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

fn print_summary(report: &SimulationReport) {
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

fn print_runtime_hint(error: &SimulationError) {
    match error {
        SimulationError::Tokenizer(_) | SimulationError::EmptyInput => {
            eprintln!(
                "Hint: gunakan hanya karakter a-z dan spasi, misalnya --input \"spiking brain\""
            );
        }
        _ => {}
    }
}

fn help_text() -> &'static str {
    "\
rekayasa_nural_brain

Usage:
  cargo run -- [options]

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
  --help, -h                      Tampilkan bantuan
  --version, -V                   Tampilkan versi

Examples:
  cargo run --
  cargo run -- --input \"brain plasticity\"
  cargo run -- --interactive --tick-ms 60
"
}

fn display_char(character: char) -> char {
    if character == ' ' { '_' } else { character }
}
