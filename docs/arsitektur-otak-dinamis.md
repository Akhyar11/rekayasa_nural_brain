# Arsitektur Otak Dinamis

## Tujuan

Mode `chat` dirancang sebagai fondasi `dynamic persistent brain`, bukan model fixed-size. Struktur internal bisa berkembang berdasarkan interaksi:

- token sensor bertambah saat karakter baru muncul
- token kata dipromosikan saat pola kata cukup sering muncul
- token frasa dipromosikan saat n-gram berulang melewati ambang
- context node dibuat saat pola konteks cukup stabil
- edge diperkuat saat pola berulang dan dipangkas saat lemah/stale

## Komponen Inti

### 1. Persistent Brain State

State disimpan ke file binary, default di:

```text
.brain/brain_state.bin
```

Isi state mencakup:

- konfigurasi pertumbuhan
- token adaptif
- node graph
- edge transition dan context
- pola konteks
- memori exact prompt-response hasil training pair
- memori input terbaru

### 2. Adaptive Tokenizer

Tokenizer tidak dibekukan dari awal. Perilakunya:

- input baru selalu bisa diterima melalui token sensor tingkat karakter
- kata berulang dipromosikan menjadi token kata
- frasa berulang dipromosikan menjadi token frasa
- tokenisasi memakai greedy longest-match terhadap token yang sudah dipelajari

### 3. Growing Graph

Graph berisi:

- `Sensor` node untuk karakter
- `Lexical` node untuk kata
- `Phrase` node untuk frasa
- `Context` node untuk pola konteks berulang

Hubungan yang dipelajari:

- `Transition`: urutan token ke token
- `ContextInput`: token pembentuk konteks ke context node
- `ContextPrediction`: context node ke token lanjutan yang diprediksi

### 3A. Prompt-Response Memory

Untuk bootstrap awal yang lebih stabil, mode `train` juga menyimpan memori langsung:

- exact prompt
- kandidat response
- frekuensi kemunculan response untuk prompt tersebut

Saat prompt yang sama muncul lagi, model akan memprioritaskan memori ini sebelum jatuh ke prediksi graph. Ini sengaja dipakai agar training file benar-benar berguna untuk mengurangi jawaban ngawur pada fase awal.

Untuk dataset `id_personachat`, pasangan training diekstrak dari:

- prompt = elemen terakhir `history`
- response = elemen terakhir `candidates`

Ini dipilih karena pada format dataset tersebut kandidat terakhir adalah respons target untuk utterance terkait.

### 4. Growth Controller

Keputusan berkembang dibuat oleh model melalui statistik interaksi:

- frekuensi kemunculan permukaan token
- frekuensi n-gram
- frekuensi pola konteks
- kekuatan edge
- umur aktivasi terakhir

### 5. Pruning

Agar graph tidak tumbuh liar:

- edge lama dilemahkan dengan decay
- edge lemah dan jarang aktif dihapus
- context node tanpa edge aktif dan sudah stale dihapus

## Batas Realistis

Arsitektur ini adalah fondasi yang kuat untuk `incremental adaptive memory`, tetapi belum sama dengan LLM modern. Saat ini keluaran model masih berbasis:

- continuation dari graph
- recall pola konteks
- asosiasi dari input yang sudah dipelajari

Untuk menjadi sistem yang lebih kuat lagi, tahap lanjutan yang paling bernilai adalah:

- embedding atau similarity layer
- scoring prediction yang lebih kaya
- compression/merge node konseptual
- benchmark kualitas interaksi
- format checkpoint binary yang lebih terkompresi lagi jika ukuran state sudah besar
