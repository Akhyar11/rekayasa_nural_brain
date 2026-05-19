# Panduan Operasional

## Ringkasan

Project ini menyediakan simulasi PSCM yang sekarang bisa dijalankan dalam dua mode:

- Mode ringkas default untuk penggunaan cepat dan output stabil.
- Mode interaktif untuk observasi tick-by-tick di terminal.

## Prasyarat

- Rust toolchain stabil
- Terminal untuk mode interaktif

## Menjalankan

```bash
cargo run --
```

Contoh dengan input khusus:

```bash
cargo run -- --input "spiking brain"
```

Mode interaktif:

```bash
cargo run -- --interactive --tick-ms 60
```

## Argumen CLI

- `--input <text>`: hanya mendukung karakter `a-z` dan spasi.
- `--steps-per-char <n>`: jumlah tick aktif per karakter.
- `--context-neurons <n>`: jumlah neuron konteks di lapisan L2/3.
- `--hippocampus-capacity <n>`: kapasitas buffer episodik.
- `--replay-epochs <n>`: jumlah epoch replay saat tidur.
- `--lr-ltp <value>` dan `--lr-ltd <value>`: learning rate plastisitas.
- `--decay <value>`: decay neuromodulator antara `0.0` dan `1.0`.
- `--interactive`: aktifkan dashboard simulasi.
- `--tick-ms <n>`: delay antar tick untuk mode interaktif.
- `--no-ansi`: nonaktifkan clear-screen ANSI.

## Sifat Output

- Mode default menampilkan ringkasan hasil simulasi dan koneksi terkuat.
- Mode interaktif menampilkan status L4, L2/3, error prediksi, dan keadaan neuromodulator pada setiap tick.

## Validasi Input

Runtime sekarang gagal cepat untuk kasus berikut:

- input kosong
- karakter selain `a-z` dan spasi
- `steps-per-char = 0`
- `replay-epochs = 0`
- kapasitas hipokampus atau neuron konteks bernilai nol
- learning rate atau decay tidak valid
