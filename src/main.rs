mod tokenizer;
mod neuron;
mod synapse;
mod neuromodulator;
mod cortical;
mod memory;

use tokenizer::SpikingTokenizer;
use cortical::CorticalColumn;
use neuromodulator::Neuromodulator;
use memory::{Hippocampus, MemoryConsolidator};
use std::thread::sleep;
use std::time::Duration;

fn main() {
    // Teks ANSI untuk estetika visual premium di terminal Linux
    println!("\x1b[1;36m================================================================================\x1b[0m");
    println!("\x1b[1;32m      PREDICTIVE SPIKING COGNITIVE MODEL (PSCM) - SIMULASI SENSORY & MEMORY\x1b[0m");
    println!("\x1b[1;36m================================================================================\x1b[0m");
    println!("Menginisialisasi sistem...");

    // 1. Inisialisasi komponen dasar
    let tokenizer = SpikingTokenizer::new();
    // 27 input neuron (a-z + space) -> 8 neuron konteks L2/3
    let mut column = CorticalColumn::new(27, 8);
    let mut nm = Neuromodulator::new(0.85); // Peluruhan neuromodulator 15% per tick
    let mut hippocampus = Hippocampus::new(500);
    let consolidator = MemoryConsolidator::new();

    // Parameter pembelajaran STDP
    let lr_ltp = 0.05;
    let lr_ltd = 0.02;

    // 2. Input teks simulasi
    let input_text = "spiking brain";
    let steps_per_char = 8; // Setiap karakter disimulasikan selama 8 ticks
    let spike_matrix = tokenizer.encode(input_text, steps_per_char);
    let total_ticks = spike_matrix.len();

    println!("\x1b[34m[SISTEM]\x1b[0m Input Teks: \x1b[1;33m\"{}\"\x1b[0m", input_text);
    println!("\x1b[34m[SISTEM]\x1b[0m Total Durasi Simulasi: \x1b[1;33m{} ticks\x1b[0m", total_ticks);
    println!("Memulai Fase Aktif (Bangun / Mengamati Sensorik)...");
    sleep(Duration::from_millis(1500));

    // 3. Loop Simulasi Fase Aktif (Bangun)
    for t in 0..total_ticks {
        // Bersihkan layar terminal untuk animasi dashboard yang dinamis
        print!("\x1b[2J\x1b[H");

        // Identifikasi karakter sensorik yang sedang masuk
        let char_idx = t / steps_per_char;
        let current_char = input_text.chars().nth(char_idx).unwrap_or(' ');

        // Jalankan satu langkah kolom kortikal
        column.step(&spike_matrix[t], t, &mut nm, lr_ltp, lr_ltd);

        // Rekam pola penembakan neuron L2/3 ke dalam Hipokampus
        let l23_spikes: Vec<bool> = column.l23_neurons.iter().map(|n| n.has_spiked).collect();
        // Jika ada neuron L2/3 yang menembak, catat di Hipokampus
        if l23_spikes.iter().any(|&s| s) {
            hippocampus.record(l23_spikes);
        }

        // Lakukan peluruhan senyawa neuromodulator
        nm.step();

        // ---- DRAW PREMIUM TERMINAL DASHBOARD ----
        println!("\x1b[1;36m┌──────────────────────────────────────────────────────────────────────────────┐\x1b[0m");
        println!(
            "\x1b[1;36m│\x1b[0m \x1b[1;32mPSCM ACTIVE SENSORY SIMULATION DASHBOARD\x1b[0m                                     \x1b[1;36m│\x1b[0m"
        );
        println!("\x1b[1;36m├──────────────────────────────────────────────────────────────────────────────┤\x1b[0m");
        println!(
            "\x1b[1;36m│\x1b[0m Waktu Simulasi : Tick \x1b[1;33m{:03}/{:03}\x1b[0m  |  Karakter Masuk: [\x1b[1;35m {}\x1b[0m ]                   \x1b[1;36m│\x1b[0m",
            t, total_ticks, current_char
        );
        println!("\x1b[1;36m├──────────────────────────────────────────────────────────────────────────────┤\x1b[0m");

        // Tampilkan aktivitas Spikes L4 (Bottom-Up)
        print!("\x1b[1;36m│\x1b[0m \x1b[1mSpikes L4 (Sensory):\x1b[0m ");
        for idx in 0..27 {
            let n_char = tokenizer.decode_char(idx);
            if column.l4_neurons[idx].has_spiked {
                print!("\x1b[1;31m{}\x1b[0m", n_char);
            } else if spike_matrix[t][idx] {
                print!("\x1b[33m*\x1b[0m"); // Calon spike
            } else {
                print!(".");
            }
        }
        println!("             \x1b[1;36m│\x1b[0m");

        // Tampilkan Aktivitas Neuron L2/3 (Konteks Global)
        println!("\x1b[1;36m│\x1b[0m                                                                              \x1b[1;36m│\x1b[0m");
        println!(
            "\x1b[1;36m│\x1b[0m \x1b[1mStatus Lapisan Neokorteks Atas (L2/3 Context):\x1b[0m                               \x1b[1;36m│\x1b[0m"
        );
        for j in 0..column.num_l23 {
            let neuron = &column.l23_neurons[j];
            let spike_symbol = if neuron.has_spiked {
                "\x1b[1;32m⚡ SPIKE\x1b[0m"
            } else if neuron.refractory_steps_left > 0 {
                "\x1b[31m▓ REFRA\x1b[0m"
            } else {
                "░ IDLE "
            };

            // Bar visual untuk potensial membran (V)
            let mut v_bar = String::new();
            let bar_len = (neuron.v * 10.0).max(0.0).min(10.0) as usize;
            for _ in 0..bar_len {
                v_bar.push('█');
            }
            for _ in bar_len..10 {
                v_bar.push(' ');
            }

            println!(
                "\x1b[1;36m│\x1b[0m   Neuron #{:02} [{}] V: [{}\x1b[0m] Thres: {:.2}                     \x1b[1;36m│\x1b[0m",
                j, spike_symbol, v_bar, neuron.v_threshold
            );
        }

        println!("\x1b[1;36m├──────────────────────────────────────────────────────────────────────────────┤\x1b[0m");
        println!(
            "\x1b[1;36m│\x1b[0m \x1b[1mKondisi Kimiawi Otak & Eror Kognitif:\x1b[0m                                         \x1b[1;36m│\x1b[0m"
        );
        let error_level = column.current_prediction_error;
        let mut error_bar = String::new();
        let err_len = (error_level * 5.0).min(15.0) as usize;
        for _ in 0..err_len {
            error_bar.push('█');
        }
        for _ in err_len..15 {
            error_bar.push('░');
        }

        println!(
            "\x1b[1;36m│\x1b[0m   Eror Prediksi Lokal: [\x1b[1;31m{}\x1b[0m] ({:.1})                                  \x1b[1;36m│\x1b[0m",
            error_bar, error_level
        );
        println!(
            "\x1b[1;36m│\x1b[0m   Neuromodulator     : {}                      \x1b[1;36m│\x1b[0m",
            nm.status_string()
        );
        println!(
            "\x1b[1;36m│\x1b[0m   Memori Hipokampus  : \x1b[1;35m{:>3}\x1b[0m pola tersimpan                                    \x1b[1;36m│\x1b[0m",
            hippocampus.episodic_buffer.len()
        );
        println!("\x1b[1;36m└──────────────────────────────────────────────────────────────────────────────┘\x1b[0m");

        // Lambatkan langkah agar terlihat efek animasi di terminal
        sleep(Duration::from_millis(180));
    }

    println!("\x1b[1;32m[SUKSES]\x1b[0m Fase Aktif selesai!");
    println!("\x1b[1;34m[INFO]\x1b[0m Hipokampus telah merekam \x1b[1;35m{} pola sensorik temporal\x1b[0m.", hippocampus.episodic_buffer.len());
    println!("Mempersiapkan Fase Tidur (Sleep-Replay Consolidation)...");
    sleep(Duration::from_millis(3000));

    // 4. Jalankan Fase Replay / Konsolidasi Memori Jangka Panjang (Tidur)
    print!("\x1b[2J\x1b[H");
    println!("\x1b[1;35m================================================================================\x1b[0m");
    println!("\x1b[1;35m                  FASE TIDUR (OFF-LINE MEMORY CONSOLIDATION)\x1b[0m");
    println!("\x1b[1;35m================================================================================\x1b[0m");
    println!("Mensimulasikan gelombang tidur lambat (Slow-Wave Sleep)...");
    println!("Memutar ulang pola aktivitas Hipokampus ke Neokorteks...");
    sleep(Duration::from_millis(1500));

    let replay_ticks = consolidator.consolidate(&mut column, &mut nm, &mut hippocampus, 10);

    println!("\x1b[1;32m[SUKSES]\x1b[0m Konsolidasi selesai!");
    println!("Telah melakukan \x1b[1;33m{} ticks\x1b[0m replay asinkron di Neokorteks.", replay_ticks);
    println!("Semua pola dari Hipokampus telah ditransfer dan dihapus.");
    println!("\x1b[1;36m--------------------------------------------------------------------------------\x1b[0m");

    // 5. Analisis Hasil Konsolidasi Bobot
    println!("\x1b[1;32m[ANALISIS]\x1b[0m Peta Kekuatan Sinapsis Neokorteks Jangka Panjang (L4 <-> L2/3):");
    
    // Cari sinapsis terkuat untuk memberikan bukti empiris pembelajaran
    column.feedforward_synapses.sort_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap());
    column.feedback_synapses.sort_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap());

    println!("\n\x1b[1mTop 3 Koneksi Feedforward (L4 -> L2/3) Terkuat (Asosiasi Fitur ke Konteks):\x1b[0m");
    for i in 0..3 {
        let syn = &column.feedforward_synapses[i];
        let ch = tokenizer.decode_char(syn.pre_idx);
        println!(
            "  - Input '\x1b[1;33m{}\x1b[0m' (Neuron #{:02}) ──► Konteks #{:02}  |  Bobot Sinaptik: \x1b[1;32m{:.2}\x1b[0m",
            ch, syn.pre_idx, syn.post_idx, syn.weight
        );
    }

    println!("\n\x1b[1mTop 3 Koneksi Feedback (L2/3 -> L4) Terkuat (Sinyal Prediktif Konteks ke Fitur):\x1b[0m");
    for i in 0..3 {
        let syn = &column.feedback_synapses[i];
        let ch = tokenizer.decode_char(syn.post_idx);
        println!(
            "  - Konteks #{:02} ──► Prediksi '\x1b[1;33m{}\x1b[0m' (Neuron #{:02})  |  Bobot Sinaptik: \x1b[1;32m{:.2}\x1b[0m",
            syn.pre_idx, ch, syn.post_idx, syn.weight
        );
    }

    println!("\x1b[1;36m================================================================================\x1b[0m");
    println!("Inisialisasi dan pengujian prototipe PSCM Rust berhasil diselesaikan!");
    println!("\x1b[1;36m================================================================================\x1b[0m");
}
