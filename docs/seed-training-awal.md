# Seed Training Awal

## Tujuan

File seed awal TSV dipakai sebagai alternatif ringan bila Anda tidak ingin langsung melatih dari dataset besar.

Default `train` utama sekarang memakai:

```text
training/id_personachat/id_personachat.json
```

Sedangkan seed TSV manual ada di:

```text
training/bootstrap_seed.tsv
```

## Format

Setiap baris:

```text
prompt<TAB>response
```

Aturan:

- baris kosong diabaikan
- baris yang diawali `#` dianggap komentar
- prompt dan response wajib terisi

## Menjalankan

```bash
cargo run -- train --file training/bootstrap_seed.tsv
```

Atau file khusus:

```bash
cargo run -- train --file training/bootstrap_seed.tsv
```

## Isi Seed yang Disarankan

Untuk bootstrap yang sehat, isi file sebaiknya mencakup:

- sapaan dasar
- identitas model
- aturan kejujuran saat tidak tahu
- gaya jawaban yang diinginkan
- preferensi atau fakta personal yang ingin diingat
- domain kerja utama yang paling sering Anda pakai

## Cara Memperluas

Tambahkan pasangan yang:

- sering benar-benar Anda tanyakan
- punya jawaban stabil
- tidak ambigu
- tidak membutuhkan pengetahuan dunia yang sering berubah

Contoh yang bagus:

```text
siapa kamu	saya adalah brain yang sedang belajar dari interaksi dengan anda.
bagaimana jika kamu tidak tahu	jika saya belum tahu, saya akan mengatakannya dan tidak mengarang.
```

Contoh yang buruk:

```text
apa berita terbaru	hari ini semua baik baik saja.
```

Contoh buruk seperti itu justru menanamkan kebiasaan mengarang.
