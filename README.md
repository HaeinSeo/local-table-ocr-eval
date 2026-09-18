# Table Image Recognition Benchmark

Table-image recognition benchmark/evaluation pipeline for comparing OCR/VLM table extraction outputs against HTML ground truth.

🙌 Click https://haeinseo.github.io/local-table-ocr-eval/?v=bcd8e82

## What This Repository Contains

- Common dataset loaders and evaluators in `table_benchmark/`
- Benchmark configuration in `configs/benchmark.yaml`
- TEDS, TEDS-S, cell F1, numeric-cell accuracy, failure-rate, and seconds/table reporting
- A mobile-friendly HTML result viewer with Korean/English conclusions

The local benchmark data, third-party model repositories, and generated run outputs are intentionally ignored by git. See `README_TABLE_BENCHMARK.md` for the full local setup and execution notes.

## Static Result Viewer

For GitHub Pages, use:

```text
docs/index.html
```

This file is a portable single-file viewer with embedded display images, GT tables, predictions, rankings, preprocessing notes, and Korean/English conclusions.

## Local Regeneration

```powershell
cd E:\DU_TABLE
python -m table_benchmark.cli --config E:/DU_TABLE/configs/benchmark.yaml dataset-viewer --dataset claude_artifact_person
python -m table_benchmark.cli --config E:/DU_TABLE/configs/benchmark.yaml dataset-viewer --dataset claude_artifact_person --portable
copy E:\DU_TABLE\runs\table_benchmark_current\claude_artifact_person\model_viewer_portable.html E:\DU_TABLE\docs\index.html
```
