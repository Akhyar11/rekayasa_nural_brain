# Panduan Operasional

## Ringkasan

Project ini sekarang punya dua mode utama:

- `chat`: mode utama untuk brain dinamis yang belajar dan menyimpan state.
- `simulate`: mode eksperimen untuk simulator PSCM fixed-size.

## Prasyarat

- Rust toolchain stabil
- Terminal untuk mode `chat` interaktif atau `simulate --interactive`

## Menjalankan

Mode chat interaktif:

```bash
cargo run -- chat
```

Mode chat satu kali lalu simpan state:

```bash
cargo run -- chat --prompt "saya suka kopi"
```

Lihat state yang tersimpan:

```bash
cargo run -- inspect
```

Jalankan simulator lama:

```bash
cargo run -- simulate --input "spiking brain"
```

## Mode Chat

Default file state:

```text
.brain/brain_state.bin
```

Perintah REPL:

- `/stats`: tampilkan ringkasan graph saat ini
- `/save`: simpan state secara manual
- `/quit` atau `/exit`: keluar dan simpan state

Argumen penting:

- `--state <path>`: lokasi file state
- `--prompt <text>`: jalankan satu interaksi tanpa masuk REPL
- `--word-threshold <n>`: ambang promosi kata
- `--phrase-threshold <n>`: ambang promosi frasa
- `--context-threshold <n>`: ambang pembuatan context node
- `--max-ngram <n>`: panjang frasa maksimum
- `--context-window <n>`: panjang context window maksimum
- `--prune-interval <n>`: interval pruning graph
- `--edge-decay <value>`: decay edge lama
- `--min-edge-strength <value>`: ambang edge lemah
- `--response-token-limit <n>`: panjang maksimum respons yang dihasilkan
- `--memory-window <n>`: jumlah input terakhir yang disimpan

## Mode Inspect

Mode `inspect` memuat file state lalu menampilkan:

- jumlah interaksi
- total token
- komposisi sensor/word/phrase token
- jumlah context node
- jumlah edge
- token terbaru
- edge terkuat

## Mode Simulate

Mode ini mempertahankan simulator PSCM lama.

Argumen penting:

- `--input <text>`
- `--steps-per-char <n>`
- `--context-neurons <n>`
- `--hippocampus-capacity <n>`
- `--replay-epochs <n>`
- `--lr-ltp <value>`
- `--lr-ltd <value>`
- `--decay <value>`
- `--interactive`
- `--tick-ms <n>`
- `--no-ansi`

## Validasi dan Persistence

Mode `chat` akan:

- membuat state baru jika file belum ada
- memuat state lama jika file sudah ada
- menyimpan ulang state setelah setiap interaksi

Runtime akan gagal cepat untuk:

- input kosong
- konfigurasi growth yang tidak valid
- file state rusak atau versi state tidak cocok

Mode `simulate` akan gagal cepat untuk:

- input kosong
- karakter selain `a-z` dan spasi
- `steps-per-char = 0`
- `replay-epochs = 0`
- kapasitas hipokampus atau neuron konteks bernilai nol
- learning rate atau decay tidak valid
