# rekayasa_nural_brain

Prototype Rust untuk dua mode kerja:

- `chat`: dynamic persistent brain yang belajar dari interaksi dan menyimpan state.
- `train`: bootstrap awal dari file `prompt -> response`.
- `simulate`: simulator PSCM fixed-size untuk eksperimen arsitektur lama.

Dokumentasi utama ada di [docs/README.md](docs/README.md).

## Jalankan

Bootstrap brain dengan seed awal:

```bash
cargo run -- train
```

Default `train` sekarang memakai dataset:

```text
training/id_personachat/id_personachat.json
```

Selama `train`, CLI sekarang menampilkan progress bar, checkpoint periodik, throughput, dan ringkasan pertumbuhan state. Setiap checkpoint juga langsung menyimpan state `.bin`, dan preparasi dataset dipercepat dengan `rayon`.

Bootstrap dari file sendiri:

```bash
cargo run -- train --file training/bootstrap_seed.tsv
```

Monitoring lebih detail:

```bash
cargo run -- train --verbose --log-every 25
```

Setelah itu baru chat:

```bash
cargo run -- chat
```

Satu interaksi lalu simpan state:

```bash
cargo run -- chat --prompt "saya suka kopi"
```

Lihat ringkasan state brain:

```bash
cargo run -- inspect
```

Jalankan simulator lama:

```bash
cargo run -- simulate --input "spiking brain"
```

## Verifikasi

```bash
cargo fmt --check
cargo test
cargo clippy --all-targets --all-features -- -D warnings
```
