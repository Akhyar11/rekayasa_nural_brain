# Temuan Audit dan Penguatan

## Temuan Awal

- Logika simulasi sebelumnya terkunci di `main.rs`, sehingga sulit diuji dan dipakai ulang.
- Tidak ada `docs/` walaupun proyek menginstruksikan dokumentasi di sana.
- Mode jalan sebelumnya sepenuhnya interaktif dengan `sleep`, sehingga tidak cocok untuk automation atau validasi cepat.
- Fase konsolidasi tidur mewarisi state neuron dari fase aktif, yang membuat replay tidak dimulai dari kondisi bersih.
- Tidak ada test otomatis.
- `clippy -D warnings` gagal.

## Penguatan yang Diterapkan

- Core simulasi dipindahkan ke API library yang reusable dan testable.
- Ditambahkan CLI dengan subcommand `chat`, `inspect`, dan `simulate`.
- Ditambahkan subcommand `train` untuk bootstrap brain dari TSV manual maupun dataset `id_personachat`.
- Ditambahkan mode interaktif opsional.
- Konsolidasi memori sekarang melakukan reset state neuron sebelum dan sesudah replay.
- Hipokampus memakai `VecDeque` untuk FIFO yang lebih tepat.
- Ditambahkan test unit untuk tokenizer, neuron, sinapsis, memori, kolom kortikal, dan simulasi.
- Ditambahkan workflow CI untuk `fmt`, `test`, dan `clippy`.
- Ditambahkan `dynamic persistent brain` yang memiliki checkpoint binary, tokenizer adaptif, graph node/edge yang bisa tumbuh, context node, dan pruning.
- Ditambahkan exact prompt-response memory agar seed training awal benar-benar menanamkan respons dasar yang stabil.

## Status Siap Pakai

Project ini sekarang siap dipakai sebagai fondasi eksperimen brain dinamis yang persisten sekaligus tetap menyediakan simulator terminal lama. Ini masih belum siap untuk klaim model bahasa biologis skala produksi karena:

- belum ada benchmark performa
- belum ada dataset training/evaluation formal
- belum ada metrik kualitas prediksi bahasa
- respons masih berbasis continuation graph dan belum memakai representasi semantik yang dalam

Untuk tahap berikutnya, fokus yang paling bernilai adalah benchmark, evaluasi berbasis corpus, similarity/embedding layer, dan kompresi graph konseptual.
