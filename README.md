# rekayasa_nural_brain

Prototype Rust untuk dua mode kerja:

- `chat`: dynamic persistent brain yang belajar dari interaksi dan menyimpan state.
- `simulate`: simulator PSCM fixed-size untuk eksperimen arsitektur lama.

Dokumentasi utama ada di [docs/README.md](docs/README.md).

## Jalankan

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
