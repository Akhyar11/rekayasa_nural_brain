# Arsitektur Otak Dinamis

Implementasi terbaru tidak lagi berhenti di `next-token continuation`. Lapisan respons sekarang memakai pola `language action selection`: sistem membangkitkan beberapa kandidat respons, mengevaluasi konsistensi dan kecocokan memori, lalu memilih jalur bahasa yang paling stabil.

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
- sensory memory
- working memory
- episodic memory
- procedural memory
- neuromodulator state

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
- `ConceptMember`: anggota token ke concept node untuk soft semantic traversal

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

### 4. Brain-Native Memory Systems

Mode `chat` sekarang memisahkan memori menjadi beberapa subsistem:

- `sensory_memory`: buffer input mentah terbaru
- `working_memory`: token, konsep aktif, intent heuristik, predicted outcomes, predicted procedures, goal aktif, tone heuristik, dan confidence interaksi saat ini
- `episodic_memory`: pasangan prompt-response yang benar-benar pernah dijalani sistem
- `procedural_memory`: kebiasaan aksi bahasa yang membawa reward intrinsik tinggi
- `procedure_schemas`: prosedur eksplisit yang telah dipelajari, misalnya `procedure:addition`
- `semantic cortex`: graph token/context/concept yang tumbuh dinamis

Pemisahan ini membuat pembelajaran tidak tergantung pada satu tabel besar `prompt -> response`.

### 5. Basal Ganglia Style Action Selection

Respons tidak lagi dipilih hanya dengan meneruskan token satu per satu. Pipeline sekarang:

```text
prompt
→ exact recall / similar prompt / episodic recall / graph continuation
→ self-evaluation
→ pilih kandidat aksi bahasa terbaik
→ keluarkan teks
```

Setiap kandidat dinilai dengan kombinasi:

- kecocokan konteks
- kecocokan episodik
- keselarasan semantik
- novelty
- confidence
- reward historis
- prediction mismatch penalty

Nilai ini dimodulasi lagi oleh `dopamine`, `acetylcholine`, dan `serotonin-like stability`.

Untuk konteks yang cocok dengan prosedur yang sudah dipelajari, selector sekarang juga bisa memunculkan jalur `procedural reasoning`. Ini penting agar sistem tidak jatuh kembali ke hafalan contoh saat sebenarnya sudah punya prosedur yang dapat digeneralisasi.

### 5A. Procedural Generalization

Sistem sekarang mulai membentuk `procedure schema` dari pola aritmetika sederhana:

```text
2 + 3 = 5
4 + 2 = 6
```

akan memperkuat konsep:

- `procedure:addition`
- relasi operator `+`
- kemampuan menyelesaikan kasus baru seperti `5 + 6`

Jalur ini memakai prosedur yang diterapkan ulang, bukan exact recall dari contoh.

### 6. Local Plasticity

Sesudah aksi dipilih:

- jalur prompt → respons terpilih diperkuat
- respons yang kalah dilemahkan sedikit pada edge awalnya
- reward tinggi memperbarui procedural memory
- keberhasilan prosedur memperbarui `procedure_schemas`
- episode disimpan ke episodic memory

Ini menjaga aturan belajar tetap lokal, bukan gradient global.

### 7. Replay / Consolidation

Secara periodik, episode bernilai tinggi di-replay:

- transisi respons diperkuat ulang
- bridge prompt → respons diperkuat ulang
- exact prompt-response memory bisa distabilkan jika reward internal melewati ambang

Replay ini berfungsi sebagai `sleep consolidation` ringan di mode `chat`.

### 7A. Predictive Processing

Sebelum finalisasi respons, brain sekarang menyimpan:

- predicted outcomes
- predicted procedures
- prediction error

`prediction error` dipakai sebagai sinyal mismatch untuk:

- menurunkan reward intrinsik saat respons kurang konsisten
- menaikkan `acetylcholine` saat kejutan atau mismatch tinggi
- membedakan goal yang terselesaikan vs masih unresolved di working memory

### 8. Growth Controller

Keputusan berkembang dibuat oleh model melalui statistik interaksi:

- frekuensi kemunculan permukaan token
- frekuensi n-gram
- frekuensi pola konteks
- kekuatan edge
- umur aktivasi terakhir

### 9. Pruning

Agar graph tidak tumbuh liar:

- edge lama dilemahkan dengan decay
- edge lemah dan jarang aktif dihapus
- context node tanpa edge aktif dan sudah stale dihapus

## Batas Realistis

Arsitektur ini sekarang lebih dekat ke `brain-native cognitive engine`, tetapi tetap belum sama dengan sistem bahasa modern skala besar. Keterbatasan utama saat ini:

- intent inference masih heuristik
- semantic memory masih ditopang graph simbolik, belum embedding konseptual penuh
- reward masih intrinsik, belum punya loop evaluasi eksternal yang kaya
- replay masih sinkron di runtime chat, belum idle scheduler terpisah

Tahap lanjut yang paling bernilai:

- evaluasi reward eksternal atau human feedback lokal
- pembentukan intent/action columns yang lebih eksplisit
- replay asinkron saat idle
- semantic compression atau merge concept node
- benchmark kualitas percakapan berbasis tugas
