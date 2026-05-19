# rekayasa_nural_brain

Prototype `Predictive Spiking Cognitive Model (PSCM)` berbasis Rust untuk simulasi tokenizer spike, kolom kortikal, neuromodulator, dan konsolidasi memori.

Dokumentasi operasional dan hasil audit ada di [docs/README.md](docs/README.md).

## Jalankan

```bash
cargo run --
```

Mode interaktif:

```bash
cargo run -- --interactive --tick-ms 60
```

## Verifikasi

```bash
cargo fmt --check
cargo test
cargo clippy --all-targets --all-features -- -D warnings
```
