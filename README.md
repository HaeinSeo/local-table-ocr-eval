# Table Image Recognition Benchmark

Table-image recognition benchmark/evaluation pipeline for comparing OCR/VLM table extraction outputs against HTML ground truth.

🙌 Click https://haeinseo.github.io/local-table-ocr-eval/?v=bcd8e82

---

## 📊 Results

**`claude_artifact_person` · 18 tables · 20 model/configuration runs**

[Result viewer](docs/index.html) · generated **2026-09-18 14:28:18**

Results are grouped into **Base Runs** and **Preprocessed Runs**.
The model rankings exclude the `gt_self` check.

Each run contains 18 tables. Scores are shown to four decimal places and
sorted by mean TEDS within each group. ↑ means higher is better; ↓ means lower
is better.

### Base Runs

These runs have no explicit preprocessing profile recorded. The long-side
column lists recorded resize settings; `—` means the setting is not recorded.

| Model / run | Long side (px) | TEDS ↑ | TEDS-S ↑ | Cell F1 ↑ | Numeric ↑ | Run fail ↓ | Conv. fail ↓ | Sec/table ↓ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-VL `30b-a3b-instruct-q4_K_M` | 1536 | 0.8949 | 0.9168 | 0.5215 | 0.5303 | 0.0000 | 0.0000 | 49.5979 |
| Qwen3.5 `9b` | 1536 | 0.8910 | 0.9259 | 0.7104 | 0.7407 | 0.0000 | 0.0000 | 37.5945 |
| GLM-OCR `latest` | — | 0.8518 | 0.8980 | 0.6627 | 0.8264 | 0.0000 | 0.0000 | 10.2308 |
| Mistral Small 3.2 `24b` | — | 0.8324 | 0.8617 | 0.5886 | 0.5972 | 0.0000 | 0.0000 | 174.0756 |
| Qwen3.6 `35b` | — | 0.8294 | 0.8711 | 0.6513 | 0.6787 | 0.0000 | 0.0000 | 30.7897 |
| Mistral Small 3.1 `24b` | — | 0.8275 | 0.8635 | 0.3860 | 0.3979 | 0.0000 | 0.0000 | 188.4139 |
| Ministral 3 `14b` | 1536* | 0.7878 | 0.8197 | 0.5041 | 0.5205 | 0.0556 | 0.0000 | 116.9486 |
| Qwen2.5-VL `32b` | — | 0.7745 | 0.8255 | 0.3065 | 0.3313 | 0.0000 | 0.0000 | 346.8470 |
| Gemma 3 `12b-it-qat` | 1536 | 0.7215 | 0.8185 | 0.2521 | 0.2136 | 0.0000 | 0.0000 | 33.1380 |
| PaddleOCR-VL 1.6 `AuditAid/PaddleOCR-VL-1.6-0.9B:latest` | 1536 | 0.6998 | 0.7492 | 0.5718 | 0.6690 | 0.0000 | 0.0000 | 6.5743 |
| PaddleOCR-VL 1.5 `hf.co/PaddlePaddle/PaddleOCR-VL-1.5-GGUF` | 1536 | 0.6920 | 0.7344 | 0.5285 | 0.6225 | 0.0000 | 0.0000 | 6.8709 |
| `vllm_dots_mocr_table_v1_4096`** | — | 0.6141 | 0.9109 | 0.8038 | 0.8254 | 0.0000 | 0.0000 | 27.3218 |
| MiniCPM-V `8b` | — | 0.4228 | 0.6446 | 0.1884 | 0.2163 | 0.0000 | 0.0000 | 30.8184 |
| Qwen3-VL `8b` | — | 0.1994 | 0.2083 | 0.1968 | 0.1940 | 0.0000 | 0.7222 | 49.9009 |
| LLaVA `7b` | — | 0.0473 | 0.2266 | 0.0026 | 0.0115 | 0.0000 | 0.1667 | 25.5234 |

\* The resize setting is recorded for successful samples; the failed sample
has no resize metadata.

\*\* Stored model tag: `model`.

### Preprocessed Runs

| Model | Profile | Long side (px) | TEDS ↑ | TEDS-S ↑ | Cell F1 ↑ | Numeric ↑ | Run fail ↓ | Conv. fail ↓ | Sec/table ↓ |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-VL `30b-a3b-instruct-q4_K_M` | `doc` | 1800 | 0.9045 | 0.9277 | 0.5915 | 0.6029 | 0.0000 | 0.0000 | 48.9495 |
| Qwen3-VL `30b-a3b-instruct-q4_K_M` | `table` | 1800 | 0.8959 | 0.9145 | 0.4756 | 0.4955 | 0.0000 | 0.0000 | 46.8302 |
| GLM-OCR `latest` | `doc` | 1536 | 0.8870 | 0.9320 | 0.6440 | 0.7986 | 0.0000 | 0.0000 | 19.4372 |
| GLM-OCR `latest` | `table` | 1536 | 0.8324 | 0.8787 | 0.6433 | 0.8066 | 0.0000 | 0.0556 | 18.9805 |
| PaddleOCR-VL 1.6 `AuditAid/PaddleOCR-VL-1.6-0.9B:latest` | `doc` | 1536 | 0.7022 | 0.7426 | 0.6202 | 0.7174 | 0.0000 | 0.0000 | 6.4219 |

### Reported Metrics

Each run reports:

| Metric | Description |
| :--- | :--- |
| TEDS | HTML tree similarity, including table structure and cell text. |
| TEDS-S | Structure-only tree similarity, with cell text ignored. |
| Cell F1 | Position-wise cell-text F1 after expanding merged cells and normalizing whitespace and case. |
| Numeric | Numeric-cell accuracy at the same grid position after numeric normalization. |
| Run fail | Failed or missing executions divided by the number of samples. |
| Conv. fail | Successful executions without convertible table output divided by the number of samples. |
| Sec/table | Mean recorded processing time in seconds per table. |

The result tables use the viewer's per-table averages. For Numeric,
`summary.csv` and `comparison.csv` use pooled correct/total numeric-cell
counts instead of the mean of per-table scores.

---

## 📁 Data

**`claude_artifact_person`** contains table images paired with HTML ground truth.

The local benchmark data, third-party model repositories, and generated run outputs are intentionally ignored by git.

[`configs/benchmark.yaml`](configs/benchmark.yaml) also defines entries for
`pubtabnet` and `omnidocbench_tables`. The results above are for
`claude_artifact_person` only.

Default local paths:

```yaml
root: E:/DU_TABLE
output_dir: E:/DU_TABLE/runs/table_benchmark_current
```

---

## ⚙️ Experiment Configuration

Dataset paths, model/parser targets, and prompt templates are defined in
[`configs/benchmark.yaml`](configs/benchmark.yaml). Recorded model tags,
preprocessing profiles, and resize settings are listed with the results.

### Recorded Prompt IDs

Recorded prompt IDs are listed below. Runs without a recorded prompt ID remain
unspecified. The descriptions are summaries; full prompt text is in
[`configs/benchmark.yaml`](configs/benchmark.yaml).

| Prompt ID | Prompt summary |
| :--- | :--- |
| `html_table_v1` | Return only the HTML table grid; exclude captions and notes outside the grid; preserve rows, columns, headers, empty cells, and visible merged cells. |
| `paddleocr_vl_table_v1` | `Table Recognition:` |
| `dots_mocr_table_v1` | `Parse the document image. Return the table as HTML.` |

### Preprocessing Profiles

Preprocessed runs use the following profiles:

| Profile | Processing |
| :--- | :--- |
| `doc` | Crop near-white margins, denoise, improve local contrast, and sharpen text. |
| `table` | Apply document cleanup and reinforce horizontal and vertical table lines. |

---

## 🧩 Pipeline

```text
Table image
    │
    ▼
Preprocessing / resizing (when configured)
    │
    ▼
OCR / vision-language model
    │
    ▼
HTML table conversion
    │
    ▼
Predicted HTML
    │
    ▼
Evaluation ◄──────── HTML ground truth
    │
    ▼
Scores + failure/time records
    │
    ▼
CSV summaries + HTML viewer
```

HTML ground truth is used for evaluation, not as model input.

[`table_benchmark/cli.py`](table_benchmark/cli.py) handles data preparation and
import, model execution, table conversion, scoring, and result-viewer generation.

### What This Repository Contains

- Common dataset loaders and evaluators in `table_benchmark/`
- Benchmark configuration in `configs/benchmark.yaml`
- TEDS, TEDS-S, cell F1, numeric-cell accuracy, failure-rate, and seconds/table reporting
- A mobile-friendly HTML result viewer with Korean/English conclusions

### Repository Layout

```text
local-table-ocr-eval/
  README.md
  configs/
    benchmark.yaml
  table_benchmark/
    __init__.py
    cli.py
  docs/
    index.html
  .github/workflows/
    pages.yml
```

---

## 🔬 Local Regeneration

Regenerate the result viewer from existing local run outputs in
**`E:\DU_TABLE`**:

```powershell
cd E:\DU_TABLE
python -m table_benchmark.cli --config E:/DU_TABLE/configs/benchmark.yaml dataset-viewer --dataset claude_artifact_person
python -m table_benchmark.cli --config E:/DU_TABLE/configs/benchmark.yaml dataset-viewer --dataset claude_artifact_person --portable
copy E:\DU_TABLE\runs\table_benchmark_current\claude_artifact_person\model_viewer_portable.html E:\DU_TABLE\docs\index.html
```

---

## 📦 Static Result Viewer

For GitHub Pages, use:

```text
docs/index.html
```

This file is a portable single-file viewer with embedded display images, GT tables, predictions, rankings, preprocessing notes, and Korean/English conclusions.

The deployment workflow in
[`.github/workflows/pages.yml`](.github/workflows/pages.yml) publishes `docs/`
on a push to `main` or a manual workflow dispatch.

[📊 Interactive Results](https://haeinseo.github.io/local-table-ocr-eval/?v=bcd8e82)
· [💻 GitHub](https://github.com/HaeinSeo/local-table-ocr-eval)
