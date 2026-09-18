from __future__ import annotations

import argparse
import base64
import csv
import fnmatch
import html as html_lib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable
from urllib import parse, request

import yaml
from bs4 import BeautifulSoup
from lxml import etree, html
from apted import APTED, Config
from apted.helpers import Tree


TABLE_VIEW_CSS = """
body{font-family:Segoe UI,Arial,sans-serif;margin:16px;background:#fff;color:#111}
table{border-collapse:collapse;border-spacing:0;max-width:100%;background:#fff}
th,td{border:1px solid #555;padding:4px 6px;vertical-align:top;min-width:24px}
th{background:#f2f4f7;font-weight:600}
thead th{background:#e8edf5}
caption{caption-side:top;text-align:left;font-weight:600;margin-bottom:8px}
.empty{color:#b00020}
pre{white-space:pre-wrap;border:1px solid #ddd;padding:12px;background:#fafafa}
""".strip()


def display_table_html(table_html: str) -> str:
    if not table_html:
        body = "<p class='empty'>No table HTML extracted.</p>"
    else:
        soup = BeautifulSoup(table_html, "html.parser")
        table = soup.find("table")
        body = str(table) if table else f"<pre>{html_lib.escape(table_html)}</pre>"
    return f"<!doctype html><html><head><meta charset='utf-8'><style>{TABLE_VIEW_CSS}</style></head><body>{body}</body></html>"


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def upsert_jsonl_by_sample(path: Path, row: dict[str, Any]) -> None:
    sample_id = row.get("sample_id")
    rows = [r for r in load_jsonl(path) if r.get("sample_id") != sample_id]
    rows.append(row)
    write_jsonl(path, rows)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def command_path(candidates: Iterable[str]) -> str | None:
    for name in candidates:
        found = shutil.which(name)
        if found:
            return found
    return None


def run_text(args: list[str], timeout: int = 20, cwd: Path | None = None) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout,
        )
        return proc.returncode, (proc.stdout + proc.stderr).strip()
    except Exception as exc:
        return 999, repr(exc)


def git_head(path: Path) -> str | None:
    if not path.exists():
        return None
    code, out = run_text(["git", "-c", f"safe.directory={path.as_posix()}", "-C", str(path), "rev-parse", "HEAD"], timeout=10)
    return out.strip() if code == 0 else None


def html_from_pubtabnet(record: dict[str, Any]) -> str:
    struct = record["html"]["structure"]["tokens"]
    cells = record["html"].get("cells", [])
    cell_i = 0
    out = ["<html><body><table>"]
    for tok in struct:
        if tok == "</td>" or tok == "</th>":
            if cell_i < len(cells):
                out.extend(cells[cell_i].get("tokens", []))
                cell_i += 1
            out.append(tok)
        else:
            out.append(tok)
    out.append("</table></body></html>")
    return "".join(out)


@dataclass
class Sample:
    dataset: str
    sample_id: str
    image: str
    gt_html: str
    meta: dict[str, Any]


def load_pubtabnet(name: str, cfg: dict[str, Any]) -> list[Sample]:
    image_dir = Path(cfg["image_dir"])
    ann = Path(cfg["annotation"])
    split = cfg.get("split", "val")
    image_files = {p.name: p for p in image_dir.glob("*.png")}
    samples: list[Sample] = []
    seen: set[str] = set()
    with ann.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("split") != split:
                continue
            fn = rec["filename"]
            seen.add(fn)
            samples.append(
                Sample(
                    dataset=name,
                    sample_id=Path(fn).stem,
                    image=str(image_dir / fn),
                    gt_html=html_from_pubtabnet(rec),
                    meta={"filename": fn, "imgid": rec.get("imgid"), "split": rec.get("split")},
                )
            )
    missing_images = sorted(seen - set(image_files))
    extra_images = sorted(set(image_files) - seen)
    for s in samples:
        s.meta["missing_image"] = not Path(s.image).exists()
    return samples, {"gt_count": len(samples), "image_count": len(image_files), "missing_images": missing_images, "extra_images": extra_images}


def load_omnidocbench_tables(name: str, cfg: dict[str, Any]) -> tuple[list[Sample], dict[str, Any]]:
    root = Path(cfg["root"])
    manifest_path = Path(cfg["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    revision_path = Path(cfg.get("revision", root / "revision.txt"))
    revision = revision_path.read_text(encoding="utf-8").strip() if revision_path.exists() else None
    samples: list[Sample] = []
    missing: list[str] = []
    for rec in manifest:
        image = root / rec["image"]
        gt = root / rec["gt"]
        if not image.exists() or not gt.exists():
            missing.append(rec.get("id", str(image)))
            continue
        samples.append(
            Sample(
                dataset=name,
                sample_id=rec["id"],
                image=str(image),
                gt_html=gt.read_text(encoding="utf-8"),
                meta={**rec, "dataset_revision": revision},
            )
        )
    return samples, {"manifest_count": len(manifest), "usable_count": len(samples), "missing": missing, "revision": revision}


def select_samples(samples: list[Sample], count: int, seed: int) -> list[Sample]:
    ordered = sorted(samples, key=lambda s: s.sample_id)
    rng = random.Random(seed)
    if len(ordered) <= count:
        return ordered
    idx = sorted(rng.sample(range(len(ordered)), count))
    return [ordered[i] for i in idx]


def prepare(cfg: dict[str, Any]) -> None:
    out = ensure_dir(Path(cfg["output_dir"]))
    all_reports = {}
    for name, ds_cfg in cfg["datasets"].items():
        if ds_cfg["type"] == "pubtabnet":
            samples, report = load_pubtabnet(name, ds_cfg)
        elif ds_cfg["type"] == "omnidocbench_tables":
            samples, report = load_omnidocbench_tables(name, ds_cfg)
        else:
            raise ValueError(ds_cfg["type"])
        fixed = select_samples(samples, int(cfg["sample_count"]), int(cfg["sample_seed"]))
        ds_dir = ensure_dir(out / name)
        sample_path = ds_dir / "samples.jsonl"
        if sample_path.exists():
            sample_path.unlink()
        for s in fixed:
            append_jsonl(sample_path, {"dataset": s.dataset, "sample_id": s.sample_id, "image": s.image, "gt_html": s.gt_html, "meta": s.meta})
        report["sample_count"] = len(fixed)
        report["sample_ids"] = [s.sample_id for s in fixed]
        all_reports[name] = report
    write_json(out / "data_report.json", all_reports)


class TableTree(Tree):
    def __init__(self, tag: str, colspan: int | None = None, rowspan: int | None = None, content: list[str] | None = None, *children: Any):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = list(children)


def levenshtein(a: list[str], b: list[str]) -> int:
    try:
        import Levenshtein

        return Levenshtein.distance("".join(a), "".join(b))
    except Exception:
        import distance

        return distance.levenshtein(a, b)


class TEDSConfig(Config):
    def maximum(self, *seqs: list[str]) -> int:
        return max([len(s) for s in seqs] + [1])

    def normalized_distance(self, *seqs: list[str]) -> float:
        return float(levenshtein(seqs[0], seqs[1])) / self.maximum(*seqs)

    def rename(self, node1: TableTree, node2: TableTree) -> float:
        if node1.tag != node2.tag or node1.colspan != node2.colspan or node1.rowspan != node2.rowspan:
            return 1.0
        if node1.tag in ("td", "th") and (node1.content or node2.content):
            return self.normalized_distance(node1.content or [], node2.content or [])
        return 0.0


class TEDSEvaluator:
    def __init__(self, structure_only: bool = False, ignore_nodes: list[str] | None = None):
        self.structure_only = structure_only
        self.ignore_nodes = ignore_nodes

    def tokenize(self, node: etree._Element, out: list[str]) -> None:
        out.append(f"<{node.tag}>")
        if node.text:
            out.extend(list(node.text))
        for child in list(node):
            self.tokenize(child, out)
        out.append(f"</{node.tag}>")
        if node.tag not in ("td", "th") and node.tail:
            out.extend(list(node.tail))

    def load_tree(self, node: etree._Element, parent: TableTree | None = None) -> TableTree:
        tag = "td" if node.tag == "th" else node.tag
        if tag == "td":
            content: list[str] = []
            if not self.structure_only:
                self.tokenize(node, content)
                content = content[1:-1]
            new = TableTree(tag, int(node.attrib.get("colspan", "1")), int(node.attrib.get("rowspan", "1")), content)
        else:
            new = TableTree(tag)
        if parent is not None:
            parent.children.append(new)
        if tag != "td":
            for child in list(node):
                self.load_tree(child, new)
        return new

    def table_node(self, value: str) -> etree._Element | None:
        if not value:
            return None
        parser = html.HTMLParser(remove_comments=True, encoding="utf-8")
        try:
            root = html.fromstring(value, parser=parser)
        except Exception:
            return None
        tables = root.xpath("//table")
        if not tables:
            return None
        table = tables[0]
        if self.ignore_nodes:
            etree.strip_tags(table, *self.ignore_nodes)
        return table

    def evaluate(self, pred: str, true: str) -> float:
        pred_node = self.table_node(pred)
        true_node = self.table_node(true)
        if pred_node is None or true_node is None:
            return 0.0
        n_nodes = max(len(pred_node.xpath(".//*")), len(true_node.xpath(".//*")), 1)
        dist = APTED(self.load_tree(pred_node), self.load_tree(true_node), TEDSConfig()).compute_edit_distance()
        return max(0.0, 1.0 - float(dist) / n_nodes)


def extract_table_html(raw: str) -> tuple[str, bool, str | None]:
    if not raw:
        return "", False, "empty_output"
    text = raw.strip()
    fence = re.search(r"```(?:html)?\s*(.*?)```", text, flags=re.I | re.S)
    if fence:
        text = fence.group(1).strip()
    soup = BeautifulSoup(text, "html.parser")
    table = soup.find("table")
    if table:
        return f"<html><body>{str(table)}</body></html>", True, None
    md = markdown_table_to_html(text)
    if md:
        return md, True, "markdown_table_no_merge"
    loose_md = loose_markdown_table_to_html(text)
    if loose_md:
        return loose_md, True, "loose_markdown_table_no_merge"
    paddle = paddle_token_table_to_html(text)
    if paddle:
        return paddle, True, "paddle_token_table_no_merge"
    coord = coordinate_ocr_to_html(text)
    if coord:
        return coord, True, "coordinate_ocr_no_merge"
    return "", False, "no_table_html"


CAPTION_OR_NOTE_RE = re.compile(
    r"^(?:"
    r"(?:table|figure)\s*\d+[\.:]?\s+|"
    r"표\s*\d+[\.:]?\s+|"
    r"caption\s*:|"
    r"source\s*:|"
    r"from\s+|"
    r"note[s]?\s*:|"
    r"\*+\s*(?:p|P)\s*[<≤=]|"
    r"(?:[a-z]|\*)\s+(?:derived|from|p\s*[<≤=])|"
    r"derived\s+from|"
    r"NA,\s+|N\.?A\.?,\s+|"
    r"(?:mating|matting|fertility|gestation|birth)\s+index\s*=|"
    r"one\s+female\s+|"
    r"excludes\s+"
    r")",
    re.I,
)


def row_colspan(tr: Any) -> int:
    total = 0
    for cell in tr.find_all(["td", "th"], recursive=False):
        try:
            total += max(1, int(cell.get("colspan", 1)))
        except ValueError:
            total += 1
    return total


def row_text(tr: Any) -> str:
    return normalize_text(" ".join(tr.stripped_strings))


def is_caption_or_note_row(tr: Any, max_cols: int) -> bool:
    cells = tr.find_all(["td", "th"], recursive=False)
    nonempty = [cell for cell in cells if normalize_text(" ".join(cell.stripped_strings))]
    if len(nonempty) != 1:
        return False
    text = row_text(tr)
    if not text:
        return False
    spans_most_of_table = row_colspan(tr) >= max(1, max_cols - 1)
    return spans_most_of_table and bool(CAPTION_OR_NOTE_RE.match(text))


def strip_caption_and_note_rows(table_html: str) -> tuple[str, int]:
    if not table_html:
        return table_html, 0
    soup = BeautifulSoup(table_html, "html.parser")
    table = soup.find("table")
    if table is None:
        return table_html, 0
    removed = 0
    for caption in table.find_all("caption"):
        caption.decompose()
        removed += 1
    rows = table.find_all("tr")
    max_cols = max((row_colspan(tr) for tr in rows), default=1)
    while rows and is_caption_or_note_row(rows[0], max_cols):
        rows[0].decompose()
        removed += 1
        rows = table.find_all("tr")
    while rows and is_caption_or_note_row(rows[-1], max_cols):
        rows[-1].decompose()
        removed += 1
        rows = table.find_all("tr")
    return f"<html><body>{str(table)}</body></html>", removed


def markdown_table_to_html(text: str) -> str | None:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("|") and ln.strip().endswith("|")]
    if len(lines) < 2 or not re.match(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$", lines[1]):
        return None
    rows = []
    for line in [lines[0], *lines[2:]]:
        cells = [clean_cell_text(c) for c in line.strip("|").split("|")]
        rows.append(cells)
    html_rows = []
    for r, row in enumerate(rows):
        tag = "th" if r == 0 else "td"
        html_rows.append("<tr>" + "".join(f"<{tag}>{html_lib.escape(c)}</{tag}>" for c in row) + "</tr>")
    return "<html><body><table>" + "".join(html_rows) + "</table></body></html>"


def loose_markdown_table_to_html(text: str) -> str | None:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.count("|") >= 2:
            lines.append(stripped)
    if not lines:
        return None
    rows = []
    for line in lines:
        cells = [clean_cell_text(c) for c in line.strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-{3,}:?", c.replace(" ", "")) for c in cells):
            continue
        if len(cells) >= 2:
            rows.append(cells)
    if not rows:
        return None
    width = max(len(row) for row in rows)
    html_rows = []
    for r, row in enumerate(rows):
        tag = "th" if r == 0 else "td"
        padded = row + [""] * (width - len(row))
        html_rows.append("<tr>" + "".join(f"<{tag}>{html_lib.escape(c)}</{tag}>" for c in padded) + "</tr>")
    return "<html><body><table>" + "".join(html_rows) + "</table></body></html>"


def clean_latex_text(text: str) -> str:
    text = text.replace(r"\(\pm\)", "±")
    text = re.sub(r"\\\((.*?)\\\)", r"\1", text)
    text = re.sub(r"_\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"\^\{([^{}]+)\}", r"\1", text)
    text = text.replace("\\", "")
    return clean_cell_text(text)


def paddle_token_table_to_html(text: str) -> str | None:
    if "<fcel>" not in text or "<nl>" not in text:
        return None
    token_re = re.compile(r"<(?:fcel|ecel|lcel|ucel)>|<nl>")
    rows: list[list[str]] = []
    current: list[str] = []
    pending_token: str | None = None
    pos = 0
    for m in token_re.finditer(text):
        between = text[pos : m.start()]
        if pending_token:
            current.append(clean_latex_text(between) if pending_token == "<fcel>" else "")
        token = m.group(0)
        if token == "<nl>":
            if current:
                rows.append(current)
            current = []
            pending_token = None
        else:
            pending_token = token
        pos = m.end()
    if pending_token:
        current.append(clean_latex_text(text[pos:]) if pending_token == "<fcel>" else "")
    if current:
        rows.append(current)
    rows = [row for row in rows if any(cell for cell in row)]
    if len(rows) < 2:
        return None
    width = max(len(row) for row in rows)
    html_rows = []
    for r, row in enumerate(rows):
        padded = row + [""] * (width - len(row))
        tag = "th" if r == 0 else "td"
        html_rows.append("<tr>" + "".join(f"<{tag}>{html_lib.escape(cell)}</{tag}>" for cell in padded) + "</tr>")
    return "<html><body><table>" + "".join(html_rows) + "</table></body></html>"


def coordinate_ocr_to_html(text: str) -> str | None:
    items = []
    for line in text.splitlines():
        m = re.match(r"^(.*?)\s*\[\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]\]\s*$", line.strip())
        if not m:
            continue
        label = clean_cell_text(m.group(1))
        if not label:
            continue
        x1, y1, x2, y2 = map(int, m.groups()[1:])
        if CAPTION_OR_NOTE_RE.match(label):
            continue
        items.append({"text": label, "x": (x1 + x2) / 2, "y": (y1 + y2) / 2, "h": max(1, y2 - y1)})
    if len(items) < 4:
        return None
    median_h = sorted(i["h"] for i in items)[len(items) // 2]
    y_tol = max(8, median_h * 0.75)
    rows: list[list[dict[str, Any]]] = []
    for item in sorted(items, key=lambda i: (i["y"], i["x"])):
        if not rows or abs(rows[-1][0]["y"] - item["y"]) > y_tol:
            rows.append([item])
        else:
            rows[-1].append(item)
    rows = [sorted(row, key=lambda i: i["x"]) for row in rows if row]
    candidate_rows = [row for row in rows if len(row) >= 2]
    if len(candidate_rows) < 2:
        return None
    x_centers = []
    for row in candidate_rows:
        if len(row) >= 3:
            x_centers.extend(i["x"] for i in row)
    if len(x_centers) < 2:
        x_centers = [i["x"] for row in candidate_rows for i in row]
    x_tol = max(24, (max(x_centers) - min(x_centers)) / 24) if len(x_centers) > 1 else 40
    cols: list[float] = []
    for x in sorted(x_centers):
        if not cols or abs(cols[-1] - x) > x_tol:
            cols.append(x)
        else:
            cols[-1] = (cols[-1] + x) / 2
    if len(cols) < 2:
        return None
    html_rows = []
    for r, row in enumerate(rows):
        cells = [""] * len(cols)
        for item in row:
            col = min(range(len(cols)), key=lambda idx: abs(cols[idx] - item["x"]))
            cells[col] = (cells[col] + " " + item["text"]).strip() if cells[col] else item["text"]
        if sum(1 for c in cells if c) < 2 and r not in (0, 1):
            continue
        tag = "th" if r == 0 else "td"
        html_rows.append("<tr>" + "".join(f"<{tag}>{html_lib.escape(c)}</{tag}>" for c in cells) + "</tr>")
    if len(html_rows) < 2:
        return None
    return "<html><body><table>" + "".join(html_rows) + "</table></body></html>"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def clean_cell_text(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"</\s*$", "", text)
    text = re.sub(r"<\s*$", "", text)
    text = re.sub(r"</\s*[A-Za-z][^>]*$", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_number(text: str) -> str | None:
    t = normalize_text(text)
    if not re.search(r"\d", t):
        return None
    t = t.replace(",", "")
    t = t.replace("−", "-").replace("–", "-").replace("—", "-")
    t = re.sub(r"\s+", "", t)
    m = re.fullmatch(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))(?:%|[A-Za-z/%°μ]+)?", t)
    if not m:
        return t
    try:
        return str(Decimal(m.group(1)).normalize())
    except InvalidOperation:
        return m.group(1)


def table_grid(table_html: str) -> list[list[str | None]]:
    parser = html.HTMLParser(remove_comments=True, encoding="utf-8")
    try:
        root = html.fromstring(table_html, parser=parser)
    except Exception:
        return []
    tables = root.xpath("//table")
    if not tables:
        return []
    grid: list[list[str | None]] = []
    spans: dict[tuple[int, int], str] = {}
    for r, tr in enumerate(tables[0].xpath(".//tr")):
        row: list[str | None] = []
        c = 0
        for cell in tr.xpath("./th|./td"):
            while spans.get((r, c)) is not None:
                row.append(spans[(r, c)])
                c += 1
            text = normalize_text("".join(cell.itertext()))
            rs = int(cell.attrib.get("rowspan", "1"))
            cs = int(cell.attrib.get("colspan", "1"))
            for dc in range(cs):
                row.append(text if dc == 0 else text)
                for dr in range(1, rs):
                    spans[(r + dr, c + dc)] = text
            c += cs
        while spans.get((r, c)) is not None:
            row.append(spans[(r, c)])
            c += 1
        grid.append(row)
    return grid


def numeric_accuracy(pred_html: str, gt_html: str) -> tuple[float | None, int, int]:
    gt = table_grid(gt_html)
    pred = table_grid(pred_html)
    total = 0
    correct = 0
    for r, row in enumerate(gt):
        for c, value in enumerate(row):
            norm = normalize_number(value or "")
            if norm is None:
                continue
            total += 1
            pred_value = pred[r][c] if r < len(pred) and c < len(pred[r]) else ""
            if normalize_number(pred_value or "") == norm:
                correct += 1
    return (correct / total if total else None), correct, total


def normalize_cell_text(text: str | None) -> str:
    return normalize_text(text or "").casefold()


def cell_text_f1(pred_html: str, gt_html: str) -> tuple[float, float, float, int, int, int]:
    gt = table_grid(gt_html)
    pred = table_grid(pred_html)
    gt_total = sum(len(row) for row in gt)
    pred_total = sum(len(row) for row in pred)
    max_rows = max(len(gt), len(pred))
    correct = 0
    for r in range(max_rows):
        gt_row = gt[r] if r < len(gt) else []
        pred_row = pred[r] if r < len(pred) else []
        max_cols = max(len(gt_row), len(pred_row))
        for c in range(max_cols):
            if c >= len(gt_row) or c >= len(pred_row):
                continue
            if normalize_cell_text(gt_row[c]) == normalize_cell_text(pred_row[c]):
                correct += 1
    precision = correct / pred_total if pred_total else 0.0
    recall = correct / gt_total if gt_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1, correct, pred_total, gt_total


def load_samples(output_dir: Path, dataset: str) -> list[dict[str, Any]]:
    return load_jsonl(output_dir / dataset / "samples.jsonl")


def score_run(cfg: dict[str, Any], dataset: str, run_id: str) -> None:
    out = Path(cfg["output_dir"])
    samples = load_samples(out, dataset)
    run_dir = out / dataset / "runs" / run_id
    pred_path = run_dir / "predictions.jsonl"
    preds = {r["sample_id"]: r for r in load_jsonl(pred_path)} if pred_path.exists() else {}
    teds = TEDSEvaluator(structure_only=False)
    teds_s = TEDSEvaluator(structure_only=True)
    rows = []
    for s in samples:
        pred = preds.get(s["sample_id"], {})
        html_pred = pred.get("html", "")
        conversion_ok = bool(pred.get("conversion_ok", False))
        run_ok = bool(pred.get("run_ok", False))
        run_failure = not run_ok
        conv_failure = run_ok and not conversion_ok
        score = teds.evaluate(html_pred, s["gt_html"]) if run_ok and conversion_ok else 0.0
        score_s = teds_s.evaluate(html_pred, s["gt_html"]) if run_ok and conversion_ok else 0.0
        nacc, ncorrect, ntotal = numeric_accuracy(html_pred, s["gt_html"]) if conversion_ok else (0.0, 0, 0)
        cell_precision, cell_recall, cell_f1, cell_correct, pred_cells, gt_cells = (
            cell_text_f1(html_pred, s["gt_html"]) if run_ok and conversion_ok else (0.0, 0.0, 0.0, 0, 0, 0)
        )
        rows.append(
            {
                "dataset": dataset,
                "run_id": run_id,
                "sample_id": s["sample_id"],
                "image": s["image"],
                "teds": score,
                "teds_s": score_s,
                "numeric_cell_accuracy": "" if nacc is None else nacc,
                "numeric_cells_correct": ncorrect,
                "numeric_cells_total": ntotal,
                "cell_precision": cell_precision,
                "cell_recall": cell_recall,
                "cell_f1": cell_f1,
                "cell_text_correct": cell_correct,
                "pred_cells_total": pred_cells,
                "gt_cells_total": gt_cells,
                "run_ok": run_ok,
                "conversion_ok": conversion_ok,
                "run_failure": run_failure,
                "conversion_failure": conv_failure,
                "seconds": pred.get("seconds", ""),
                "error": pred.get("error", ""),
            }
        )
    write_csv(run_dir / "scores.csv", rows)
    with (run_dir / "scores.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = summarize(rows)
    upsert_summary(out / dataset / "summary.csv", summary)
    write_viewer(run_dir / "viewer.html", samples, preds, rows)


def write_csv(path: Path, rows: list[dict[str, Any]], append: bool = False) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    exists = path.exists() and append
    with path.open("a" if append else "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerows(rows)


def upsert_summary(path: Path, summary: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = [normalize_summary_row(row) for row in csv.DictReader(f)]
    summary = normalize_summary_row(summary)
    run_key = summary.get("model/parser") or summary.get("model_parser")
    dataset_key = summary.get("dataset")
    kept = []
    for row in rows:
        row_run = row.get("model/parser") or row.get("model_parser")
        if row.get("dataset") == dataset_key and row_run == run_key:
            continue
        kept.append(row)
    kept.append(summary)
    write_csv(path, kept)


def normalize_summary_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    if "model_parser" in out and "model/parser" not in out:
        out["model/parser"] = out.pop("model_parser")
    if "tables" in out and "table count" not in out:
        out["table count"] = out.pop("tables")
    preferred = [
        "dataset",
        "model/parser",
        "backend_tag",
        "table count",
        "TEDS ↑",
        "TEDS-S ↑",
        "cell_f1 ↑",
        "numeric_cell_accuracy ↑",
        "run_failure_rate ↓",
        "conversion_failure_rate ↓",
        "seconds_per_table ↓",
    ]
    ordered = {key: out.pop(key) for key in preferred if key in out}
    ordered.update(out)
    return ordered


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    def mean_num(key: str) -> float:
        vals = [float(r[key]) for r in rows if r[key] != ""]
        return sum(vals) / len(vals) if vals else 0.0

    total_num = sum(int(r["numeric_cells_total"]) for r in rows)
    correct_num = sum(int(r["numeric_cells_correct"]) for r in rows)
    return {
        "dataset": rows[0]["dataset"] if rows else "",
        "model/parser": rows[0]["run_id"] if rows else "",
        "backend_tag": "",
        "table count": n,
        "TEDS ↑": mean_num("teds"),
        "TEDS-S ↑": mean_num("teds_s"),
        "cell_f1 ↑": mean_num("cell_f1"),
        "numeric_cell_accuracy ↑": (correct_num / total_num if total_num else ""),
        "run_failure_rate ↓": sum(1 for r in rows if r["run_failure"]) / n if n else 0.0,
        "conversion_failure_rate ↓": sum(1 for r in rows if r["conversion_failure"]) / n if n else 0.0,
        "seconds_per_table ↓": mean_num("seconds") if any(r["seconds"] != "" for r in rows) else "",
    }


def compare(cfg: dict[str, Any]) -> None:
    out = Path(cfg["output_dir"])
    rows: list[dict[str, Any]] = []
    configured = list(cfg["datasets"])
    discovered = [p.name for p in out.iterdir() if p.is_dir() and (p / "summary.csv").exists()] if out.exists() else []
    for dataset in sorted(set(configured + discovered)):
        summary_path = out / dataset / "summary.csv"
        if not summary_path.exists():
            continue
        with summary_path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                rows.append(
                    {
                        "dataset": row.get("dataset", dataset),
                        "model/parser": row.get("model/parser") or row.get("model_parser") or "",
                        "backend/tag": row.get("backend_tag", ""),
                        "table count": row.get("table count") or row.get("tables") or "",
                        "TEDS ↑": row.get("TEDS ↑", ""),
                        "TEDS-S ↑": row.get("TEDS-S ↑", ""),
                        "cell F1 ↑": row.get("cell_f1 ↑", ""),
                        "numeric cell accuracy ↑": row.get("numeric_cell_accuracy ↑", ""),
                        "run failure rate ↓": row.get("run_failure_rate ↓", ""),
                        "conversion failure rate ↓": row.get("conversion_failure_rate ↓", ""),
                        "seconds/table ↓": row.get("seconds_per_table ↓", ""),
                    }
                )
    write_csv(out / "comparison.csv", rows)


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def latest_preds_by_sample(path: Path) -> dict[str, dict[str, Any]]:
    preds: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        sample_id = row.get("sample_id")
        if sample_id:
            preds[sample_id] = row
    return preds


def image_data_uri(path: Path, max_long_side: int = 1400, quality: int = 82) -> str:
    try:
        from io import BytesIO
        from PIL import Image

        img = Image.open(path).convert("RGB")
        long_side = max(img.size)
        if long_side > max_long_side:
            scale = max_long_side / long_side
            img = img.resize((round(img.width * scale), round(img.height * scale)), Image.Resampling.LANCZOS)
        buf = BytesIO()
        img.save(buf, "JPEG", quality=quality, optimize=True, progressive=True)
        return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
    except Exception:
        suffix = path.suffix.lower()
        mime = "image/png"
        if suffix in {".jpg", ".jpeg"}:
            mime = "image/jpeg"
        elif suffix == ".webp":
            mime = "image/webp"
        return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def write_dataset_viewer(cfg: dict[str, Any], dataset: str, portable: bool = False) -> Path:
    out = Path(cfg["output_dir"])
    dataset_dir = out / dataset
    runs_dir = dataset_dir / "runs"
    samples = load_samples(out, dataset)
    runs: list[dict[str, Any]] = []
    if runs_dir.exists():
        for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
            pred_file = run_dir / "predictions.jsonl"
            score_file = run_dir / "scores.csv"
            if not pred_file.exists():
                continue
            preds = latest_preds_by_sample(pred_file)
            score_map = {r["sample_id"]: r for r in read_csv_rows(score_file) if r.get("sample_id")}
            sample_rows = []
            for s in samples:
                sid = s["sample_id"]
                pred = preds.get(sid, {})
                score = score_map.get(sid, {})
                sample_rows.append(
                    {
                        "sample_id": sid,
                        "image": image_data_uri(Path(s["image"])) if portable else Path(s["image"]).as_uri(),
                        "gt_doc": display_table_html(s["gt_html"]),
                        "pred_doc": display_table_html(pred.get("html", "")),
                        "teds": score.get("teds", ""),
                        "teds_s": score.get("teds_s", ""),
                        "cell_f1": score.get("cell_f1", ""),
                        "numeric": score.get("numeric_cell_accuracy", ""),
                        "run_ok": pred.get("run_ok", ""),
                        "conversion_ok": pred.get("conversion_ok", ""),
                        "seconds": pred.get("seconds", ""),
                        "preprocess_profile": pred.get("preprocess_profile", ""),
                        "backend_tag": pred.get("backend_tag", ""),
                        "prompt_id": pred.get("prompt_id", ""),
                        "long_image_side": pred.get("long_image_side", ""),
                        "min_image_side": pred.get("min_image_side", ""),
                        "error": pred.get("error", ""),
                        "raw_path": pred.get("raw_path", ""),
                        "html_path": pred.get("html_path", ""),
                    }
                )
            runs.append({"run_id": run_dir.name, "samples": sample_rows})
    payload = {"dataset": dataset, "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "runs": runs, "sample_ids": [s["sample_id"] for s in samples]}
    payload_json = json.dumps(payload, ensure_ascii=False)
    path = dataset_dir / ("model_viewer_portable.html" if portable else "model_viewer.html")
    ensure_dir(path.parent)
    path.write_text(
        f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Cache-Control" content="no-store, no-cache, must-revalidate, max-age=0">
<meta http-equiv="Pragma" content="no-cache">
<meta http-equiv="Expires" content="0">
<title>{html_lib.escape(dataset)} Model Viewer</title>
<style>
*{{box-sizing:border-box}}
body{{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#070707;color:#f5ead2}}
header{{position:sticky;top:0;background:#0c0c0d;border-bottom:1px solid #b9934a;padding:14px 18px;z-index:2;box-shadow:0 8px 24px rgba(0,0,0,.35)}}
main{{padding:18px;background:linear-gradient(180deg,#101010 0,#070707 260px)}}
.controls{{display:flex;gap:12px;align-items:center;flex-wrap:wrap}}
label{{color:#d9bd7a;font-weight:600;font-size:13px}}
select,button{{font:inherit;padding:7px 10px;background:#151515;color:#f7ecd3;border:1px solid #b9934a;border-radius:6px;max-width:100%}}
button{{cursor:pointer;background:#b9934a;color:#090909;font-weight:700}}
button:hover{{background:#d4af5f}}
.status{{margin-top:9px;color:#c9b27c;font-size:13px}}
.metrics{{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0 16px}}
.metric{{background:#111;border:1px solid #6f5729;border-left:4px solid #d4af5f;border-radius:7px;padding:8px 11px;color:#f7ecd3;box-shadow:0 6px 18px rgba(0,0,0,.25)}}
.grid{{display:grid;grid-template-columns:320px 1fr 1fr;gap:14px;align-items:start}}
.panel{{background:#101010;border:1px solid #6f5729;border-radius:8px;padding:11px;min-width:0;box-shadow:0 10px 28px rgba(0,0,0,.28)}}
.panel h2{{font-size:14px;margin:0 0 9px;color:#d4af5f;letter-spacing:.02em;text-transform:uppercase}}
.section-title{{font-size:13px;color:#ffd66b;margin:14px 0 6px;text-transform:uppercase;letter-spacing:.02em}}
img{{max-width:100%;border:1px solid #b9934a;background:#fff}}
iframe{{width:100%;height:560px;border:1px solid #b9934a;background:#fff;border-radius:4px}}
.error{{color:#ff9f9f;white-space:pre-wrap}}
.paths{{font-size:12px;color:#c9b27c;word-break:break-all;line-height:1.5}}
.ranking-table{{width:100%;border-collapse:collapse;margin-top:8px;font-size:13px}}
.ranking-table th,.ranking-table td{{border-bottom:1px solid #3d321c;padding:8px 9px;text-align:right}}
.ranking-table th{{color:#d4af5f;background:#14110b}}
.ranking-table td:first-child,.ranking-table th:first-child,.ranking-table td:nth-child(2),.ranking-table th:nth-child(2){{text-align:left}}
.ranking-table td{{vertical-align:middle}}
.ranking-table .run-cell{{min-width:310px;max-width:520px;text-align:left}}
.run-main{{font-weight:800;color:#fff1bd;font-size:13px;line-height:1.25}}
.run-sub{{margin-top:3px;color:#aa925b;font-size:11px;line-height:1.35;word-break:break-all;font-family:Consolas,Menlo,monospace}}
.run-tags{{display:flex;gap:5px;flex-wrap:wrap;margin-top:5px}}
.tag{{display:inline-flex;align-items:center;border:1px solid #6f5729;background:#171207;color:#f4d889;border-radius:999px;padding:2px 7px;font-size:11px;font-weight:700;white-space:nowrap}}
.tag.dim{{color:#bfa46e;background:#0e0c08}}
.ranking-table tr{{cursor:pointer}}
.ranking-table tbody tr:hover{{background:#1d180d}}
.ranking-table tbody tr.best-model{{font-weight:800;color:#ffe7a3;background:#221a08}}
.ranking-table tbody tr.best-model .run-main{{text-decoration:underline;text-decoration-thickness:2px;text-underline-offset:4px}}
.ranking-table tbody tr.best-model td:first-child{{color:#ffd66b}}
.ranking-note{{color:#c9b27c;font-size:12px;margin:0 0 8px}}
.formula-box{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:9px;margin:8px 0 12px}}
.formula{{border:1px solid #3d321c;background:#0b0b0b;border-left:3px solid #d4af5f;border-radius:6px;padding:9px 11px;color:#ead7a8;font-size:12px;line-height:1.5}}
.formula b{{display:block;color:#ffd66b;margin-bottom:4px}}
.math{{display:block;color:#fff3c4;background:#15110a;border:1px solid #4f3f1e;border-radius:4px;padding:7px 8px;margin-top:5px;overflow:auto}}
.preprocess-guide{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:10px;margin:10px 0 14px}}
.prep-card{{border:1px solid #4d3d1d;background:#0d0b08;border-radius:7px;padding:10px 12px;color:#e8d6a8;font-size:12px;line-height:1.55}}
.prep-card b{{display:inline-flex;align-items:center;color:#ffd66b;font-size:13px;margin-bottom:3px}}
.prep-card .good{{color:#f4e2ad;font-weight:700}}
.prep-card .warn{{color:#c9b27c}}
.conclusion-grid{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:8px 0 4px}}
.conclusion-card{{border:1px solid #4d3d1d;background:#0b0b0b;border-left:3px solid #d4af5f;border-radius:7px;padding:11px 13px;color:#ead7a8;line-height:1.55;font-size:13px}}
.conclusion-card b{{display:block;color:#ffd66b;margin-bottom:5px}}
.conclusion-card ul{{margin:6px 0 0;padding-left:18px}}
.conclusion-card li{{margin:4px 0}}
.table-wrap{{width:100%;overflow-x:auto}}
@media(max-width:1100px){{.grid{{grid-template-columns:1fr}} iframe{{height:420px}}}}
@media(max-width:760px){{
  header{{position:static;padding:12px}}
  main{{padding:10px}}
  .controls{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}
  .controls label{{display:flex;flex-direction:column;gap:4px;min-width:0}}
  .controls button{{min-height:38px}}
  .metrics{{display:grid;grid-template-columns:1fr 1fr;gap:7px}}
  .metric{{padding:7px 8px;font-size:12px}}
  .panel{{padding:10px;border-radius:7px}}
  iframe{{height:360px}}
  .formula-box,.preprocess-guide,.conclusion-grid{{grid-template-columns:1fr}}
  .ranking-table,.ranking-table tbody{{display:block;width:100%}}
  .ranking-table thead{{display:none}}
  .ranking-table tr{{display:grid;grid-template-columns:46px minmax(0,1fr);gap:0 10px;border:1px solid #3d321c;border-radius:8px;margin:10px 0;padding:9px;background:#0b0b0b}}
  .ranking-table tbody tr.best-model{{background:#211908;border-color:#b9934a}}
  .ranking-table td{{display:flex;align-items:center;justify-content:space-between;grid-column:1 / -1;border-bottom:1px solid #241d10;text-align:right;padding:7px 2px;position:relative;min-height:30px}}
  .ranking-table td::before{{content:attr(data-label);color:#d4af5f;text-align:left;font-weight:700;padding-right:12px}}
  .ranking-table td:first-child{{grid-column:1;grid-row:1;display:flex;align-items:center;justify-content:center;border:1px solid #6f5729;border-radius:999px;background:#15110a;color:#ffd66b;font-weight:900;min-height:34px;padding:0;text-align:center}}
  .ranking-table td:first-child::before{{display:none}}
  .ranking-table .run-cell{{grid-column:2;grid-row:1;display:block;min-width:0;max-width:none;text-align:left;padding:0 0 8px 0;border-bottom:0}}
  .ranking-table .run-cell::before{{display:none}}
  .ranking-table .run-cell + td{{border-top:1px solid #3d321c;margin-top:2px}}
  .ranking-table td:last-child{{border-bottom:0}}
  .run-main{{font-size:14px}}
  .run-sub{{font-size:10px}}
}}
</style>
<script>
window.MathJax = {{
  tex: {{inlineMath: [['\\\\(', '\\\\)']], displayMath: [['\\\\[', '\\\\]']]}},
  svg: {{fontCache: 'global'}}
}};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
</head>
<body>
<header>
  <div class="controls">
    <label>Model <select id="run"></select></label>
    <label>Sample <select id="sample"></select></label>
    <button id="prev">Prev</button>
    <button id="next">Next</button>
  </div>
  <div class="status" id="status"></div>
</header>
<main>
  <div class="panel" style="margin-bottom:14px">
    <h2>Conclusion / 결론</h2>
    <div id="conclusion" class="conclusion-grid"></div>
  </div>
  <div class="metrics" id="metrics"></div>
  <div class="grid">
    <section class="panel"><h2>Image</h2><img id="image"></section>
    <section class="panel"><h2>GT</h2><iframe id="gt"></iframe></section>
    <section class="panel"><h2>Prediction</h2><iframe id="pred"></iframe></section>
  </div>
  <div class="panel" style="margin-top:14px">
    <h2>Error / Paths</h2>
    <div class="error" id="error"></div>
    <div class="paths" id="paths"></div>
  </div>
  <div class="panel" style="margin-top:14px">
    <h2>Best Model Ranking</h2>
    <div class="formula-box">
      <div class="formula"><b>TEDS ↑</b>HTML tree similarity with structure and cell text.<span class="math">\\(\\mathrm{{TEDS}} = 1 - \\frac{{d_{{tree}}(P,G)}}{{\\max(|P|,|G|)}}\\)</span></div>
      <div class="formula"><b>TEDS-S ↑</b>Structure-only TEDS after ignoring cell text.<span class="math">\\(\\mathrm{{TEDS\\text{{-}}S}} = 1 - \\frac{{d_{{tree}}(S(P),S(G))}}{{\\max(|S(P)|,|S(G)|)}}\\)</span></div>
      <div class="formula"><b>Cell F1 ↑</b>Exact cell text match at the same expanded grid position.<span class="math">\\(F_1 = \\frac{{2PR}}{{P+R}},\\quad P=\\frac{{C}}{{N_{{pred}}}},\\quad R=\\frac{{C}}{{N_{{GT}}}}\\)</span></div>
      <div class="formula"><b>Numeric ↑</b>Numeric cell accuracy at the same grid position.<span class="math">\\(\\mathrm{{Numeric}} = \\frac{{N_{{correct\\ numeric}}}}{{N_{{GT\\ numeric}}}}\\)</span></div>
      <div class="formula"><b>Failure Rates ↓</b>Execution and HTML conversion failure rates.<span class="math">\\(\\mathrm{{RunFail}} = \\frac{{N_{{run\\ fail}}}}{{N}},\\quad \\mathrm{{ConvFail}} = \\frac{{N_{{conv\\ fail}}}}{{N}}\\)</span></div>
      <div class="formula"><b>Sec/Table ↓</b>Mean inference time per table.<span class="math">\\(\\mathrm{{Sec/Table}} = \\frac{{\\sum_i t_i}}{{N}}\\)</span></div>
    </div>
    <p class="ranking-note">Sorted by mean TEDS. Run failures, timeouts, empty outputs, and HTML conversion failures are not excluded; they stay in the denominator and receive 0 for TEDS, TEDS-S, Cell F1, and Numeric. DeepSeek-OCR Ollama runs use markdown OCR prompts; markdown cannot encode rowspan/colspan, so merged-cell structure loss is expected. GT self-check is excluded from the model ranking.</p>
    <div class="preprocess-guide">
      <div class="prep-card"><b>doc preprocessing</b><br><span class="good">OCR/text-first cleanup.</span> Crops near-white margins, denoises, boosts local contrast, and sharpens characters without intentionally thickening table grid lines. Use this as the default for report/table screenshots when small numbers, ± signs, stars, and superscripts matter.</div>
      <div class="prep-card"><b>table preprocessing</b><br><span class="good">structure/grid-first cleanup.</span> Starts from the document cleanup, then reinforces horizontal and vertical table lines. It can help when row/column borders are faint, but may hurt text and numeric recognition because thin glyphs can be over-sharpened or merged with grid lines.</div>
      <div class="prep-card"><b>How to read this split</b><br><span class="warn">Base Runs</span> are original image runs with no preprocessing. <span class="warn">Preprocessed Runs</span> are the same model family with an explicit image preprocessing profile recorded in the prediction JSONL.</div>
    </div>
    <div id="ranking"></div>
  </div>
</main>
<script>
const DATA = {payload_json};
const runSel = document.getElementById('run');
const sampleSel = document.getElementById('sample');
const statusEl = document.getElementById('status');
const metricsEl = document.getElementById('metrics');
const imgEl = document.getElementById('image');
const gtEl = document.getElementById('gt');
const predEl = document.getElementById('pred');
const errEl = document.getElementById('error');
const pathsEl = document.getElementById('paths');
const rankingEl = document.getElementById('ranking');
const conclusionEl = document.getElementById('conclusion');

function option(value, text) {{
  const o = document.createElement('option');
  o.value = value;
  o.textContent = text;
  return o;
}}

for (const run of DATA.runs) runSel.appendChild(option(run.run_id, run.run_id));
for (const sid of DATA.sample_ids) sampleSel.appendChild(option(sid, sid));

function currentRun() {{
  return DATA.runs.find(r => r.run_id === runSel.value) || DATA.runs[0];
}}

function currentSample(run) {{
  return (run?.samples || []).find(s => s.sample_id === sampleSel.value) || run?.samples?.[0];
}}

function setFrame(frame, html) {{
  frame.srcdoc = html || "<!doctype html><meta charset='utf-8'><p>No HTML</p>";
}}

function numberValue(v) {{
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}}

function boolValue(v) {{
  return v === true || v === 'True' || v === 'true' || v === 1 || v === '1';
}}

function fmt(v) {{
  const n = numberValue(v);
  return n === null ? '' : n.toFixed(4);
}}

function esc(s) {{
  return String(s ?? '').replace(/[&<>"']/g, ch => ({{
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;'
  }}[ch]));
}}

function firstValue(samples, key) {{
  const hit = (samples || []).find(s => s && s[key]);
  return hit ? hit[key] : '';
}}

function inferredPreprocess(runId, samples) {{
  const direct = firstValue(samples, 'preprocess_profile');
  return direct || '';
}}

function shortRunLabel(runId, model, preprocess, promptId) {{
  let label = model || runId;
  if (!model) {{
    label = label.replace(/^ollama_/, '').replace(/^vllm_/, '');
    if (preprocess) label = label.replace(new RegExp(`_${{preprocess}}(_|$)`), '$1');
    if (promptId) label = label.replace(`_${{promptId}}`, '');
    label = label.replace(/_/g, ':').replace(/:latest$/, ':latest');
  }}
  return label;
}}

function runCell(r) {{
  const pieces = [];
  if (r.preprocess) pieces.push(`<span class="tag">${{esc(r.preprocess)}}</span>`);
  if (r.promptId) pieces.push(`<span class="tag dim">${{esc(r.promptId)}}</span>`);
  if (r.longImageSide) pieces.push(`<span class="tag dim">long ${{esc(r.longImageSide)}}px</span>`);
  const tags = pieces.length ? `<div class="run-tags">${{pieces.join('')}}</div>` : '';
  return `<td class="run-cell"><div class="run-main">${{esc(r.displayName)}}</div><div class="run-sub">${{esc(r.run_id)}}</div>${{tags}}</td>`;
}}

function scoreText(r) {{
  return `${{esc(r.displayName)}} (TEDS ${{fmt(r.teds)}}, TEDS-S ${{fmt(r.tedsS)}}, Cell F1 ${{fmt(r.cellF1)}}, Numeric ${{fmt(r.numeric)}})`;
}}

function renderConclusion(rows) {{
  if (!rows.length) {{
    conclusionEl.innerHTML = '<div class="conclusion-card"><b>결론</b>아직 표시할 실행 결과가 없습니다.</div>';
    return;
  }}
  const basicRows = rows.filter(r => !r.preprocess);
  const preRows = rows.filter(r => r.preprocess);
  const topOverall = rows[0];
  const topBase = basicRows[0];
  const topPre = preRows[0];
  const fastestGood = rows
    .filter(r => numberValue(r.teds) !== null && numberValue(r.teds) >= 0.80)
    .sort((a,b) => (a.secs ?? 999999) - (b.secs ?? 999999))[0];
  const topNumeric = [...rows].sort((a,b) => (b.numeric ?? -1) - (a.numeric ?? -1))[0];
  const ko = `
    <div class="conclusion-card"><b>한국어 결론</b>
      <ul>
        <li>현재 최고 TEDS는 <b>${{scoreText(topOverall)}}</b>입니다.</li>
        ${{topBase ? `<li>원본 이미지 기준 최고 모델은 <b>${{scoreText(topBase)}}</b>입니다.</li>` : ''}}
        ${{topPre ? `<li>전처리 적용 기준 최고 조합은 <b>${{scoreText(topPre)}}</b>입니다. 현재 데이터에서는 doc 전처리가 표 선 강화(table)보다 문자/숫자 보존에 유리했습니다.</li>` : ''}}
        ${{topNumeric ? `<li>숫자 셀 정확도 기준으로는 <b>${{scoreText(topNumeric)}}</b>가 가장 좋습니다.</li>` : ''}}
        ${{fastestGood ? `<li>TEDS 0.80 이상 중 속도 우선 선택지는 <b>${{scoreText(fastestGood)}}</b>입니다.</li>` : ''}}
      </ul>
    </div>`;
  const en = `
    <div class="conclusion-card"><b>English Summary</b>
      <ul>
        <li>The best overall TEDS run is <b>${{scoreText(topOverall)}}</b>.</li>
        ${{topBase ? `<li>The best no-preprocessing baseline is <b>${{scoreText(topBase)}}</b>.</li>` : ''}}
        ${{topPre ? `<li>The best preprocessed run is <b>${{scoreText(topPre)}}</b>. On this dataset, doc preprocessing preserved text and numeric cells better than the grid-emphasizing table profile.</li>` : ''}}
        ${{topNumeric ? `<li>For numeric-cell accuracy, the strongest run is <b>${{scoreText(topNumeric)}}</b>.</li>` : ''}}
        ${{fastestGood ? `<li>Among runs with TEDS ≥ 0.80, the fastest practical option is <b>${{scoreText(fastestGood)}}</b>.</li>` : ''}}
      </ul>
    </div>`;
  conclusionEl.innerHTML = ko + en;
}}

function mean(values) {{
  const nums = values.map(numberValue).filter(v => v !== null);
  return nums.length ? nums.reduce((a,b) => a + b, 0) / nums.length : null;
}}

function buildRanking() {{
  const rows = DATA.runs
    .filter(run => run.run_id !== 'gt_self')
    .map(run => {{
      const samples = run.samples || [];
      const total = samples.length || 0;
      const preprocess = inferredPreprocess(run.run_id, samples);
      const model = firstValue(samples, 'backend_tag');
      const promptId = firstValue(samples, 'prompt_id');
      const longImageSide = firstValue(samples, 'long_image_side');
      const teds = mean(samples.map(s => s.teds));
      const tedsS = mean(samples.map(s => s.teds_s));
      const cellF1 = mean(samples.map(s => s.cell_f1));
      const numeric = mean(samples.map(s => s.numeric));
      const runFail = total ? samples.filter(s => !boolValue(s.run_ok)).length / total : null;
      const convFail = total ? samples.filter(s => boolValue(s.run_ok) && !boolValue(s.conversion_ok)).length / total : null;
      const secs = mean(samples.map(s => s.seconds));
      const displayName = shortRunLabel(run.run_id, model, preprocess, promptId);
      return {{run_id: run.run_id, displayName, model, promptId, longImageSide, total, preprocess, teds, tedsS, cellF1, numeric, runFail, convFail, secs}};
    }})
    .sort((a,b) => (b.teds ?? -1) - (a.teds ?? -1));
  if (!rows.length) {{
    rankingEl.textContent = 'No model runs yet.';
    return;
  }}
  const basicRows = rows.filter(r => !r.preprocess);
  const preRows = rows.filter(r => r.preprocess);
  renderConclusion(rows);
  function tableHtml(title, tableRows, includePreprocess) {{
    if (!tableRows.length) return '';
    let html = `<h3 class="section-title">${{title}}</h3>`;
    html += '<div class="table-wrap"><table class="ranking-table"><thead><tr><th>#</th><th>Model</th>';
    if (includePreprocess) html += '<th>Prep</th>';
    html += '<th>Tables</th><th>TEDS ↑</th><th>TEDS-S ↑</th><th>Cell F1 ↑</th><th>Numeric ↑</th><th>Run Fail ↓</th><th>Conv Fail ↓</th><th>Sec/Table ↓</th></tr></thead><tbody>';
    tableRows.forEach((r, i) => {{
      const klass = i === 0 ? ' class="best-model"' : '';
      html += `<tr${{klass}} data-run="${{esc(r.run_id)}}"><td data-label="#">${{i + 1}}</td>${{runCell(r)}}`;
      if (includePreprocess) html += `<td data-label="Prep"><span class="tag">${{esc(r.preprocess || '')}}</span></td>`;
      html += `<td data-label="Tables">${{r.total}}</td><td data-label="TEDS">${{fmt(r.teds)}}</td><td data-label="TEDS-S">${{fmt(r.tedsS)}}</td><td data-label="Cell F1">${{fmt(r.cellF1)}}</td><td data-label="Numeric">${{fmt(r.numeric)}}</td><td data-label="Run Fail">${{fmt(r.runFail)}}</td><td data-label="Conv Fail">${{fmt(r.convFail)}}</td><td data-label="Sec/Table">${{fmt(r.secs)}}</td></tr>`;
    }});
    html += '</tbody></table></div>';
    return html;
  }}
  let html = tableHtml('Base Runs - no preprocessing', basicRows, false);
  html += tableHtml('Preprocessed Runs - same model plus image preprocessing', preRows, true);
  rankingEl.innerHTML = html;
  rankingEl.querySelectorAll('tr[data-run]').forEach(tr => {{
    tr.addEventListener('click', () => {{
      runSel.value = tr.dataset.run;
      render();
      window.scrollTo({{top: 0, behavior: 'smooth'}});
    }});
  }});
}}

function render() {{
  const run = currentRun();
  const s = currentSample(run);
  if (!run || !s) {{
    statusEl.textContent = 'No runs found.';
    return;
  }}
  statusEl.textContent = `${{DATA.dataset}} | ${{run.run_id}} | ${{s.sample_id}} | generated ${{DATA.generated_at || ''}}`;
  metricsEl.innerHTML = '';
  const metrics = [
    ['TEDS', s.teds],
    ['TEDS-S', s.teds_s],
    ['Cell F1', s.cell_f1],
    ['Numeric', s.numeric],
    ['Run OK', s.run_ok],
    ['Conversion OK', s.conversion_ok],
    ['Seconds', s.seconds],
  ];
  for (const [k,v] of metrics) {{
    const div = document.createElement('div');
    div.className = 'metric';
    div.textContent = `${{k}}: ${{v ?? ''}}`;
    metricsEl.appendChild(div);
  }}
  imgEl.src = s.image;
  setFrame(gtEl, s.gt_doc);
  setFrame(predEl, s.pred_doc);
  errEl.textContent = s.error || '';
  pathsEl.innerHTML = `raw: ${{s.raw_path || ''}}<br>html: ${{s.html_path || ''}}`;
}}

runSel.addEventListener('change', render);
sampleSel.addEventListener('change', render);
document.getElementById('prev').addEventListener('click', () => {{
  sampleSel.selectedIndex = Math.max(0, sampleSel.selectedIndex - 1);
  render();
}});
document.getElementById('next').addEventListener('click', () => {{
  sampleSel.selectedIndex = Math.min(sampleSel.options.length - 1, sampleSel.selectedIndex + 1);
  render();
}});
buildRanking();
render();
</script>
</body>
</html>
""",
        encoding="utf-8",
    )
    return path


def wrapped_table_html(table: Any) -> str:
    return f"<html><body>{str(table)}</body></html>"


def write_artifact_dataset(dataset: str, data_dir: Path, output_dir: Path, rows: list[dict[str, Any]], source: str) -> None:
    ensure_dir(data_dir / "tables")
    ensure_dir(data_dir / "gt")
    manifest = []
    sample_path = output_dir / dataset / "samples.jsonl"
    if sample_path.exists():
        sample_path.unlink()
    for row in rows:
        manifest.append(
            {
                "id": row["id"],
                "image": f"tables/{Path(row['image']).name}",
                "gt": f"gt/{Path(row['gt']).name}",
                "source": source,
                **row.get("meta", {}),
            }
        )
        gt_html = Path(row["gt"]).read_text(encoding="utf-8")
        append_jsonl(
            sample_path,
            {
                "dataset": dataset,
                "sample_id": row["id"],
                "image": str(Path(row["image"]).resolve()),
                "gt_html": gt_html,
                "meta": {"source": source, **row.get("meta", {})},
            },
        )
    write_json(data_dir / "manifest.json", manifest)
    write_json(
        output_dir / dataset / "import_report.json",
        {"dataset": dataset, "data_dir": str(data_dir), "source": source, "sample_count": len(rows)},
    )


def image_ext_from_data_uri(src: str) -> str:
    m = re.match(r"data:image/([^;,]+)", src, flags=re.I)
    if not m:
        return ".png"
    ext = m.group(1).lower()
    return ".jpg" if ext in ("jpeg", "jpg") else f".{ext}"


def save_image_src(src: str, base_dir: Path, dest: Path) -> Path | None:
    src = src.strip()
    if src.startswith("data:image/"):
        head, encoded = src.split(",", 1)
        target = dest.with_suffix(image_ext_from_data_uri(head))
        target.write_bytes(base64.b64decode(encoded))
        return target
    parsed = parse.urlparse(src)
    if parsed.scheme in ("http", "https"):
        target = dest.with_suffix(Path(parsed.path).suffix or ".png")
        request.urlretrieve(src, target)
        return target
    if parsed.scheme == "file":
        source = Path(parse.unquote(parsed.path))
    else:
        source = (base_dir / parse.unquote(src)).resolve()
    if not source.exists() or not source.is_file():
        return None
    target = dest.with_suffix(source.suffix or ".png")
    shutil.copy2(source, target)
    return target


def import_artifact_html(source_html: Path, dataset: str, data_dir: Path, output_dir: Path) -> None:
    html_files = [source_html] if source_html.is_file() else sorted([*source_html.glob("*.html"), *source_html.glob("*.htm"), *source_html.glob("*.txt")])
    rows: list[dict[str, Any]] = []
    image_dir = ensure_dir(data_dir / "tables")
    gt_dir = ensure_dir(data_dir / "gt")
    for html_file in html_files:
        soup = BeautifulSoup(html_file.read_text(encoding="utf-8", errors="replace"), "html.parser")
        images = [img.get("src") for img in soup.find_all("img") if img.get("src")]
        tables = soup.find_all("table")
        pair_count = min(len(images), len(tables))
        for i in range(pair_count):
            sample_id = f"{dataset}_{len(rows):04d}"
            image_path = save_image_src(images[i], html_file.parent, image_dir / sample_id)
            if image_path is None:
                continue
            gt_path = gt_dir / f"{sample_id}.html"
            gt_path.write_text(wrapped_table_html(tables[i]), encoding="utf-8")
            rows.append(
                {
                    "id": sample_id,
                    "image": str(image_path),
                    "gt": str(gt_path),
                    "meta": {
                        "source_file": str(html_file),
                        "source_image_index": i,
                        "source_table_index": i,
                        "source_image_count": len(images),
                        "source_table_count": len(tables),
                    },
                }
            )
    write_artifact_dataset(dataset, data_dir, output_dir, rows, f"html:{source_html}")


def import_paired_dirs(images_dir: Path, gt_dir: Path, dataset: str, data_dir: Path, output_dir: Path) -> None:
    out_images = ensure_dir(data_dir / "tables")
    out_gt = ensure_dir(data_dir / "gt")
    gt_by_stem = {p.stem: p for p in gt_dir.glob("*.html")}
    rows: list[dict[str, Any]] = []
    for image in sorted([p for p in images_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}]):
        gt = gt_by_stem.get(image.stem)
        if not gt:
            continue
        sample_id = image.stem
        image_target = out_images / image.name
        gt_target = out_gt / f"{sample_id}.html"
        shutil.copy2(image, image_target)
        html_text = gt.read_text(encoding="utf-8", errors="replace")
        table_html, ok, _ = extract_table_html(html_text)
        gt_target.write_text(table_html if ok else html_text, encoding="utf-8")
        rows.append({"id": sample_id, "image": str(image_target), "gt": str(gt_target), "meta": {"source_image": str(image), "source_gt": str(gt)}})
    write_artifact_dataset(dataset, data_dir, output_dir, rows, f"paired:{images_dir}|{gt_dir}")


def import_artifact_json(source_json: Path, dataset: str, data_dir: Path, output_dir: Path) -> None:
    payload = json.loads(source_json.read_text(encoding="utf-8"))
    items = payload.get("items", payload if isinstance(payload, list) else [])
    image_dir = ensure_dir(data_dir / "tables")
    gt_dir = ensure_dir(data_dir / "gt")
    rows: list[dict[str, Any]] = []
    for i, item in enumerate(items):
        sample_id = str(item.get("id") or f"{dataset}_{i:04d}")
        gt_html = item.get("gt_html") or item.get("html") or item.get("table_html") or ""
        image_value = item.get("image_data_uri") or item.get("image") or item.get("src") or ""
        table_html, ok, _ = extract_table_html(gt_html)
        if not ok:
            table_html = gt_html
        image_path = save_image_src(image_value, source_json.parent, image_dir / sample_id)
        if image_path is None or not table_html:
            continue
        gt_path = gt_dir / f"{sample_id}.html"
        gt_path.write_text(table_html, encoding="utf-8")
        rows.append({"id": sample_id, "image": str(image_path), "gt": str(gt_path), "meta": {"source_json": str(source_json), "source_index": i}})
    write_artifact_dataset(dataset, data_dir, output_dir, rows, f"json:{source_json}")


def write_viewer(path: Path, samples: list[dict[str, Any]], preds: dict[str, dict[str, Any]], scores: list[dict[str, Any]]) -> None:
    score_map = {r["sample_id"]: r for r in scores}
    parts = [
        "<!doctype html><meta charset='utf-8'><title>Table Benchmark Viewer</title>",
        "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:24px} .item{border-top:1px solid #ccc;padding:20px 0} .grid{display:grid;grid-template-columns:320px 1fr 1fr;gap:16px;align-items:start} img{max-width:320px;border:1px solid #ddd} iframe{width:100%;height:320px;border:1px solid #ddd;background:white} code{white-space:pre-wrap}.bad{color:#b00020}</style>",
    ]
    for s in samples:
        sid = s["sample_id"]
        pred = preds.get(sid, {})
        sc = score_map.get(sid, {})
        gt_doc = html_lib.escape(display_table_html(s["gt_html"]))
        pred_doc = html_lib.escape(display_table_html(pred.get("html", "")))
        parts.append(f"<div class='item'><h2>{html_lib.escape(sid)}</h2>")
        parts.append(
            f"<p>TEDS={sc.get('teds','')} | TEDS-S={sc.get('teds_s','')} | numeric={sc.get('numeric_cell_accuracy','')} | error=<span class='bad'>{html_lib.escape(str(sc.get('error','')))}</span></p>"
        )
        parts.append("<div class='grid'>")
        parts.append(f"<div><h3>Image</h3><img src='{Path(s['image']).as_uri()}'></div>")
        parts.append(f"<div><h3>GT</h3><iframe srcdoc=\"{gt_doc}\"></iframe></div>")
        parts.append(f"<div><h3>Prediction</h3><iframe srcdoc=\"{pred_doc}\"></iframe></div>")
        parts.append("</div></div>")
    ensure_dir(path.parent)
    path.write_text("\n".join(parts), encoding="utf-8")


def env_report(cfg: dict[str, Any]) -> None:
    out = Path(cfg["output_dir"])
    report: dict[str, Any] = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "targets": [], "ollama": {}, "repos": {}}
    for repo in ["PubTabNet", "OmniDocBench", "opendataloader-pdf", "MinerU", "docling", "marker", "PaddleOCR", "GLM-OCR", "DeepSeek-OCR-2", "Qwen3-VL"]:
        path = Path(cfg["root"]) / repo
        report["repos"][repo] = {"path": str(path), "exists": path.exists(), "commit": git_head(path)}
    code, ollama_list = run_text(["ollama", "list"], timeout=20)
    report["ollama"]["list_code"] = code
    report["ollama"]["list"] = ollama_list
    installed_models = parse_ollama_list(ollama_list) if code == 0 else []
    for model in installed_models:
        c, show = run_text(["ollama", "show", model], timeout=20)
        report["ollama"].setdefault("show", {})[model] = {"code": c, "text": show}
    for target in cfg["targets"]:
        item = dict(target)
        if target["kind"] == "ollama_vision":
            matched = []
            if "model_tag" in target:
                matched = [m for m in installed_models if m == target["model_tag"]]
            else:
                for pat in target.get("model_patterns", []):
                    matched.extend([m for m in installed_models if fnmatch.fnmatchcase(m.lower(), pat.lower())])
            item["installed_models"] = sorted(set(matched))
            item["status"] = "available_pending_vision_probe" if matched else "not_installed"
        else:
            cmd = command_path(target.get("command_candidates", []))
            item["command_path"] = cmd
            item["status"] = "available" if cmd else "not_installed_or_not_on_path"
        report["targets"].append(item)
    write_json(out / "environment_report.json", report)


def parse_ollama_list(text: str) -> list[str]:
    models = []
    for line in text.splitlines()[1:]:
        parts = line.split()
        if parts:
            models.append(parts[0])
    return models


def run_oracle(cfg: dict[str, Any], dataset: str, run_id: str = "gt_self") -> None:
    out = Path(cfg["output_dir"])
    run_dir = ensure_dir(out / dataset / "runs" / run_id / "raw")
    pred_file = out / dataset / "runs" / run_id / "predictions.jsonl"
    if pred_file.exists():
        pred_file.unlink()
    for s in load_samples(out, dataset):
        raw = s["gt_html"]
        (run_dir / f"{s['sample_id']}.html").write_text(raw, encoding="utf-8")
        append_jsonl(
            pred_file,
            {
                "dataset": dataset,
                "sample_id": s["sample_id"],
                "target": run_id,
                "run_ok": True,
                "conversion_ok": True,
                "raw_path": str(run_dir / f"{s['sample_id']}.html"),
                "html": raw,
                "seconds": 0.0,
            },
        )
    score_run(cfg, dataset, run_id)


def run_corrupt_eval(cfg: dict[str, Any], dataset: str, run_id: str = "eval_sanity_corrupt") -> None:
    out = Path(cfg["output_dir"])
    pred_file = out / dataset / "runs" / run_id / "predictions.jsonl"
    raw_dir = ensure_dir(out / dataset / "runs" / run_id / "raw")
    if pred_file.exists():
        pred_file.unlink()
    for i, s in enumerate(load_samples(out, dataset)):
        raw = s["gt_html"]
        if i % 3 == 0:
            pred_html = re.sub(r"<td([^>]*)>.*?</td>", r"<td\1>XXX</td>", raw, count=1, flags=re.S)
        elif i % 3 == 1:
            pred_html = re.sub(r"\s(rowspan|colspan)=\"[^\"]+\"", "", raw, count=1)
        else:
            pred_html = ""
        conversion_ok = bool(pred_html)
        (raw_dir / f"{s['sample_id']}.html").write_text(pred_html, encoding="utf-8")
        append_jsonl(
            pred_file,
            {
                "dataset": dataset,
                "sample_id": s["sample_id"],
                "target": run_id,
                "run_ok": True,
                "conversion_ok": conversion_ok,
                "raw_path": str(raw_dir / f"{s['sample_id']}.html"),
                "html": pred_html,
                "seconds": 0.0,
                "error": "" if conversion_ok else "synthetic_empty_output",
            },
        )
    score_run(cfg, dataset, run_id)


def image_to_pdf(image: Path, pdf: Path) -> None:
    from PIL import Image

    ensure_dir(pdf.parent)
    img = Image.open(image).convert("RGB")
    img.save(pdf, "PDF", resolution=300.0)


def crop_near_white_border(img: Any, margin: int = 8) -> Any:
    from PIL import ImageChops, ImageOps

    gray = ImageOps.grayscale(img)
    bg = gray.point(lambda p: 255 if p > 245 else 0)
    inv = ImageChops.invert(bg)
    bbox = inv.getbbox()
    if not bbox:
        return img
    left, top, right, bottom = bbox
    left = max(0, left - margin)
    top = max(0, top - margin)
    right = min(img.width, right + margin)
    bottom = min(img.height, bottom + margin)
    if right - left < img.width * 0.5 or bottom - top < img.height * 0.5:
        return img
    return img.crop((left, top, right, bottom))


def cv2_table_preprocess(img: Any, profile: str) -> Any | None:
    try:
        import cv2
        import numpy as np
        from PIL import Image
    except Exception:
        return None

    arr = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    if profile in {"doc", "ocr", "table"}:
        gray = cv2.fastNlMeansDenoising(gray, None, 7, 7, 21)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
        sharp = cv2.addWeighted(gray, 1.65, blur, -0.65, 0)
        if profile == "table":
            binary = cv2.adaptiveThreshold(sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9)
            inv = 255 - binary
            h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
            lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, h_kernel) | cv2.morphologyEx(inv, cv2.MORPH_OPEN, v_kernel)
            reinforced = np.minimum(sharp, 255 - lines)
            return Image.fromarray(reinforced).convert("RGB")
        return Image.fromarray(sharp).convert("RGB")
    if profile in {"bw", "binary", "threshold"}:
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11)
        return Image.fromarray(binary).convert("RGB")
    return None


def prepare_vision_image(
    image: Path,
    target: Path,
    min_side: int | None = None,
    long_side: int | None = None,
    preprocess_profile: str | None = None,
) -> Path:
    profile = (preprocess_profile or "none").lower()
    if not min_side and not long_side and profile in {"", "none", "raw"}:
        return image
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps

    ensure_dir(target.parent)
    img = Image.open(image).convert("RGB")
    if profile in {"doc", "ocr", "table", "crop"}:
        img = crop_near_white_border(img)
    width, height = img.size
    scale = 1.0
    if min_side:
        short = min(width, height)
        if short < min_side:
            scale = max(scale, min_side / short)
    if long_side:
        long = max(width, height)
        if long != long_side:
            scale = long_side / long
    if scale != 1.0:
        img = img.resize((round(width * scale), round(height * scale)), Image.Resampling.LANCZOS)
    if profile in {"doc", "ocr", "table"}:
        cv_img = cv2_table_preprocess(img, profile)
        if cv_img is not None:
            img = cv_img
        else:
            img = ImageOps.grayscale(img)
            img = ImageOps.autocontrast(img, cutoff=1)
            img = ImageEnhance.Contrast(img).enhance(1.45)
            img = ImageEnhance.Sharpness(img).enhance(1.8)
            img = img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=160, threshold=2)).convert("RGB")
    elif profile in {"clean", "contrast", "sharp"}:
        img = ImageOps.grayscale(img)
        img = ImageOps.autocontrast(img, cutoff=1)
        img = ImageEnhance.Contrast(img).enhance(1.25)
        img = ImageEnhance.Sharpness(img).enhance(1.4)
        img = img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=120, threshold=3)).convert("RGB")
    elif profile in {"bw", "binary", "threshold"}:
        cv_img = cv2_table_preprocess(img, profile)
        if cv_img is not None:
            img = cv_img
        else:
            gray = ImageOps.grayscale(img)
            gray = ImageOps.autocontrast(gray, cutoff=1)
            img = gray.point(lambda p: 255 if p > 185 else 0, mode="1").convert("RGB")
    elif profile in {"gray", "grayscale"}:
        img = ImageOps.grayscale(img)
        img = ImageOps.autocontrast(img, cutoff=1).convert("RGB")
    elif profile == "crop":
        img = ImageOps.grayscale(img)
        img = ImageOps.autocontrast(img, cutoff=1).convert("RGB")
    elif profile not in {"", "none", "raw"}:
        raise ValueError(f"Unknown preprocess profile: {preprocess_profile}")
    img.save(target, "PNG")
    return target


def prepare_ollama_image(image: Path, target: Path, min_side: int | None = None, long_side: int | None = None) -> Path:
    return prepare_vision_image(image, target, min_side=min_side, long_side=long_side)


def completed_sample_ids(pred_file: Path) -> set[str]:
    done: set[str] = set()
    if not pred_file.exists():
        return done
    for row in load_jsonl(pred_file):
        if row.get("run_ok") and row.get("conversion_ok") and row.get("html"):
            done.add(row["sample_id"])
    return done


def ollama_payload(model: str, prompt: str, image_b64: str, num_predict: int | None = None, num_ctx: int | None = None) -> dict[str, Any]:
    options: dict[str, Any] = {"temperature": 0}
    if num_predict:
        options["num_predict"] = num_predict
    if num_ctx:
        options["num_ctx"] = num_ctx
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "options": options,
    }
    if re.search(r"qwen3|thinking", model, flags=re.I):
        payload["think"] = False
    return payload


def safe_run_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_").replace(":", "_")


def default_run_id(prefix: str, model: str, prompt_id: str, preprocess_profile: str | None = None) -> str:
    parts = [prefix, safe_run_part(model)]
    profile = (preprocess_profile or "").lower()
    if profile and profile not in {"none", "raw"}:
        parts.append(profile)
    parts.append(prompt_id)
    return "_".join(part for part in parts if part)


def run_ollama(
    cfg: dict[str, Any],
    dataset: str,
    model: str,
    run_id: str | None = None,
    prompt_id: str = "html_table_v1",
    timeout_seconds: int | None = None,
    num_predict: int | None = None,
    num_ctx: int | None = None,
    min_image_side: int | None = None,
    long_image_side: int | None = None,
    preprocess_profile: str | None = None,
) -> None:
    run_id = run_id or default_run_id("ollama", model, prompt_id, preprocess_profile)
    if prompt_id == "html_table_v1" and "deepseek-ocr" in model.lower():
        prompt_id = "deepseek_ocr_prompt_v1"
    if min_image_side is None and "deepseek-ocr" in model.lower():
        min_image_side = 1024
    if long_image_side is None and "deepseek-ocr" in model.lower():
        long_image_side = 1536
    if num_ctx is None and "deepseek-ocr" in model.lower():
        num_ctx = 8192
    out = Path(cfg["output_dir"])
    raw_dir = ensure_dir(out / dataset / "runs" / run_id / "raw")
    html_dir = ensure_dir(out / dataset / "runs" / run_id / "pred_html")
    prep_dir = ensure_dir(out / dataset / "runs" / run_id / "preprocessed")
    pred_file = out / dataset / "runs" / run_id / "predictions.jsonl"
    done = completed_sample_ids(pred_file)
    prompt = cfg["prompts"][prompt_id]
    timeout = int(timeout_seconds or cfg["timeout_seconds"])
    attempted = 0
    succeeded = 0
    converted = 0
    skipped = 0
    for s in load_samples(out, dataset):
        if s["sample_id"] in done:
            skipped += 1
            continue
        attempted += 1
        print(f"[{run_id}] {s['sample_id']} -> {model}", flush=True)
        start = time.perf_counter()
        try:
            image_path = prepare_vision_image(
                Path(s["image"]),
                prep_dir / f"{s['sample_id']}.png",
                min_side=min_image_side,
                long_side=long_image_side,
                preprocess_profile=preprocess_profile,
            )
            b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
            payload = json.dumps(ollama_payload(model, prompt, b64, num_predict=num_predict, num_ctx=num_ctx)).encode("utf-8")
            req = request.Request("http://localhost:11434/api/generate", data=payload, headers={"Content-Type": "application/json"})
            with request.urlopen(req, timeout=timeout) as resp:
                raw_json = json.loads(resp.read().decode("utf-8"))
            raw = raw_json.get("response", "") or ""
            thinking = raw_json.get("thinking", "") or ""
            parse_source = raw if raw.strip() else thinking
            html_pred, ok, conv_error = extract_table_html(parse_source)
            removed_caption_note_rows = 0
            if ok:
                html_pred, removed_caption_note_rows = strip_caption_and_note_rows(html_pred)
            raw_path = raw_dir / f"{s['sample_id']}.txt"
            raw_path.write_text(raw, encoding="utf-8")
            raw_json_path = raw_dir / f"{s['sample_id']}.json"
            raw_json_path.write_text(json.dumps(raw_json, ensure_ascii=False, indent=2), encoding="utf-8")
            html_path = html_dir / f"{s['sample_id']}.html"
            html_path.write_text(display_table_html(html_pred), encoding="utf-8")
            row = {"dataset": dataset, "sample_id": s["sample_id"], "target": run_id, "backend_tag": model, "prompt_id": prompt_id, "input_image_path": str(image_path), "min_image_side": min_image_side or "", "long_image_side": long_image_side or "", "preprocess_profile": preprocess_profile or "", "num_ctx": num_ctx or "", "run_ok": True, "conversion_ok": ok, "raw_path": str(raw_path), "raw_json_path": str(raw_json_path), "html_path": str(html_path), "html": html_pred, "seconds": time.perf_counter() - start, "error": conv_error or "", "removed_caption_note_rows": removed_caption_note_rows, "response_chars": len(raw), "thinking_chars": len(thinking)}
        except Exception as exc:
            row = {"dataset": dataset, "sample_id": s["sample_id"], "target": run_id, "backend_tag": model, "run_ok": False, "conversion_ok": False, "raw_path": "", "html_path": "", "html": "", "seconds": time.perf_counter() - start, "error": repr(exc)}
        upsert_jsonl_by_sample(pred_file, row)
        if row.get("run_ok"):
            succeeded += 1
        if row.get("conversion_ok") and row.get("html"):
            converted += 1
        if row.get("run_ok") and row.get("conversion_ok") and row.get("html"):
            done.add(s["sample_id"])
    score_run(cfg, dataset, run_id)
    print(
        f"[{run_id}] done: attempted={attempted}, skipped={skipped}, run_ok={succeeded}, html={converted}, output={pred_file}",
        flush=True,
    )


def extract_openai_message(data: dict[str, Any]) -> str:
    try:
        content = data["choices"][0]["message"].get("content", "")
    except Exception:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
        return "\n".join(p for p in parts if p)
    return str(content or "")


def run_openai_vision(
    cfg: dict[str, Any],
    dataset: str,
    model: str,
    run_id: str | None,
    base_url: str = "http://localhost:8000/v1",
    prompt_id: str = "ocr_table_html_v1",
    timeout_seconds: int | None = None,
    max_tokens: int = 8192,
    min_image_side: int | None = None,
    long_image_side: int | None = None,
    preprocess_profile: str | None = None,
    top_k: int | None = None,
) -> None:
    run_id = run_id or default_run_id("vllm", model, prompt_id, preprocess_profile)
    out = Path(cfg["output_dir"])
    raw_dir = ensure_dir(out / dataset / "runs" / run_id / "raw")
    html_dir = ensure_dir(out / dataset / "runs" / run_id / "pred_html")
    prep_dir = ensure_dir(out / dataset / "runs" / run_id / "preprocessed")
    pred_file = out / dataset / "runs" / run_id / "predictions.jsonl"
    done = completed_sample_ids(pred_file)
    prompt = cfg["prompts"][prompt_id]
    timeout = int(timeout_seconds or cfg["timeout_seconds"])
    attempted = succeeded = converted = skipped = 0
    endpoint = base_url.rstrip("/") + "/chat/completions"
    for s in load_samples(out, dataset):
        if s["sample_id"] in done:
            skipped += 1
            continue
        attempted += 1
        print(f"[{run_id}] {s['sample_id']} -> {model}", flush=True)
        start = time.perf_counter()
        try:
            image_path = prepare_vision_image(
                Path(s["image"]),
                prep_dir / f"{s['sample_id']}.png",
                min_side=min_image_side,
                long_side=long_image_side,
                preprocess_profile=preprocess_profile,
            )
            b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                "temperature": 0.0,
                "max_tokens": max_tokens,
            }
            if top_k is not None:
                payload["top_k"] = top_k
            req = request.Request(endpoint, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"})
            with request.urlopen(req, timeout=timeout) as resp:
                raw_json = json.loads(resp.read().decode("utf-8"))
            raw = extract_openai_message(raw_json)
            html_pred, ok, conv_error = extract_table_html(raw)
            removed_caption_note_rows = 0
            if ok:
                html_pred, removed_caption_note_rows = strip_caption_and_note_rows(html_pred)
            raw_path = raw_dir / f"{s['sample_id']}.txt"
            raw_path.write_text(raw, encoding="utf-8")
            raw_json_path = raw_dir / f"{s['sample_id']}.json"
            raw_json_path.write_text(json.dumps(raw_json, ensure_ascii=False, indent=2), encoding="utf-8")
            html_path = html_dir / f"{s['sample_id']}.html"
            html_path.write_text(display_table_html(html_pred), encoding="utf-8")
            row = {"dataset": dataset, "sample_id": s["sample_id"], "target": run_id, "backend_tag": model, "base_url": base_url, "prompt_id": prompt_id, "input_image_path": str(image_path), "preprocess_profile": preprocess_profile or "", "run_ok": True, "conversion_ok": ok, "raw_path": str(raw_path), "raw_json_path": str(raw_json_path), "html_path": str(html_path), "html": html_pred, "seconds": time.perf_counter() - start, "error": conv_error or "", "removed_caption_note_rows": removed_caption_note_rows, "response_chars": len(raw)}
        except Exception as exc:
            row = {"dataset": dataset, "sample_id": s["sample_id"], "target": run_id, "backend_tag": model, "base_url": base_url, "prompt_id": prompt_id, "run_ok": False, "conversion_ok": False, "raw_path": "", "html_path": "", "html": "", "seconds": time.perf_counter() - start, "error": repr(exc)}
        upsert_jsonl_by_sample(pred_file, row)
        if row.get("run_ok"):
            succeeded += 1
        if row.get("conversion_ok") and row.get("html"):
            converted += 1
        if row.get("run_ok") and row.get("conversion_ok") and row.get("html"):
            done.add(s["sample_id"])
    score_run(cfg, dataset, run_id)
    print(f"[{run_id}] done: attempted={attempted}, skipped={skipped}, run_ok={succeeded}, html={converted}, output={pred_file}", flush=True)


def restyle_run_html(cfg: dict[str, Any], dataset: str, run_id: str | None = None) -> None:
    out = Path(cfg["output_dir"])
    runs_root = out / dataset / "runs"
    run_dirs = [runs_root / run_id] if run_id else [p for p in runs_root.iterdir() if p.is_dir()]
    for run_dir in run_dirs:
        pred_file = run_dir / "predictions.jsonl"
        if not pred_file.exists():
            continue
        html_dir = ensure_dir(run_dir / "pred_html")
        count = 0
        for row in load_jsonl(pred_file):
            sample_id = row.get("sample_id")
            if not sample_id:
                continue
            (html_dir / f"{sample_id}.html").write_text(display_table_html(row.get("html", "")), encoding="utf-8")
            count += 1
        print(f"restyled {run_dir.name}: {count}")
        score_run(cfg, dataset, run_dir.name)


def reconvert_run(cfg: dict[str, Any], dataset: str, run_id: str) -> None:
    out = Path(cfg["output_dir"])
    run_dir = out / dataset / "runs" / run_id
    pred_file = run_dir / "predictions.jsonl"
    if not pred_file.exists():
        raise SystemExit(f"Missing predictions: {pred_file}")
    html_dir = ensure_dir(run_dir / "pred_html")
    rows = []
    converted = 0
    for row in load_jsonl(pred_file):
        raw_path = Path(row.get("raw_path", ""))
        if raw_path.exists():
            html_pred, ok, conv_error = extract_table_html(raw_path.read_text(encoding="utf-8", errors="replace"))
            removed = 0
            if ok:
                html_pred, removed = strip_caption_and_note_rows(html_pred)
            html_path = html_dir / f"{row['sample_id']}.html"
            html_path.write_text(display_table_html(html_pred), encoding="utf-8")
            row.update(
                {
                    "conversion_ok": ok,
                    "html_path": str(html_path),
                    "html": html_pred,
                    "error": conv_error or "",
                    "removed_caption_note_rows": removed,
                }
            )
            if ok:
                converted += 1
        rows.append(row)
    write_jsonl(pred_file, rows)
    score_run(cfg, dataset, run_id)
    print(f"reconverted {run_id}: {converted}/{len(rows)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="E:/DU_TABLE/configs/benchmark.yaml")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("env")
    sub.add_parser("prepare")
    sub.add_parser("compare")
    p_oracle = sub.add_parser("oracle")
    p_oracle.add_argument("--dataset", required=True)
    p_corrupt = sub.add_parser("sanity")
    p_corrupt.add_argument("--dataset", required=True)
    p_score = sub.add_parser("score")
    p_score.add_argument("--dataset", required=True)
    p_score.add_argument("--run-id", required=True)
    p_restyle = sub.add_parser("restyle")
    p_restyle.add_argument("--dataset", required=True)
    p_restyle.add_argument("--run-id")
    p_reconvert = sub.add_parser("reconvert")
    p_reconvert.add_argument("--dataset", required=True)
    p_reconvert.add_argument("--run-id", required=True)
    p_dataset_viewer = sub.add_parser("dataset-viewer")
    p_dataset_viewer.add_argument("--dataset", required=True)
    p_dataset_viewer.add_argument("--portable", action="store_true", help="Embed images as data URIs for GitHub Pages/mobile sharing.")
    p_ollama = sub.add_parser("run-ollama")
    p_ollama.add_argument("--dataset", required=True)
    p_ollama.add_argument("--model", required=True)
    p_ollama.add_argument("--run-id")
    p_ollama.add_argument("--prompt-id", default="html_table_v1")
    p_ollama.add_argument("--timeout-seconds", type=int)
    p_ollama.add_argument("--num-predict", type=int)
    p_ollama.add_argument("--num-ctx", type=int)
    p_ollama.add_argument("--min-image-side", type=int)
    p_ollama.add_argument("--long-image-side", type=int)
    p_ollama.add_argument("--preprocess-profile", choices=["none", "raw", "crop", "clean", "gray", "bw", "doc", "ocr", "table"])
    p_openai = sub.add_parser("run-openai-vision")
    p_openai.add_argument("--dataset", required=True)
    p_openai.add_argument("--model", required=True)
    p_openai.add_argument("--run-id")
    p_openai.add_argument("--base-url", default="http://localhost:8000/v1")
    p_openai.add_argument("--prompt-id", default="ocr_table_html_v1")
    p_openai.add_argument("--timeout-seconds", type=int)
    p_openai.add_argument("--max-tokens", type=int, default=8192)
    p_openai.add_argument("--min-image-side", type=int)
    p_openai.add_argument("--long-image-side", type=int)
    p_openai.add_argument("--preprocess-profile", choices=["none", "raw", "crop", "clean", "gray", "bw", "doc", "ocr", "table"])
    p_openai.add_argument("--top-k", type=int)
    p_import = sub.add_parser("import-artifact")
    p_import.add_argument("--dataset", default="claude_artifact_person")
    p_import.add_argument("--data-dir", default="E:/DU_TABLE/data/claude_artifact_person")
    p_import.add_argument("--source-html")
    p_import.add_argument("--source-json")
    p_import.add_argument("--images-dir")
    p_import.add_argument("--gt-dir")
    args = parser.parse_args()
    cfg = read_yaml(Path(args.config))
    if args.cmd == "env":
        env_report(cfg)
    elif args.cmd == "prepare":
        prepare(cfg)
    elif args.cmd == "compare":
        compare(cfg)
    elif args.cmd == "oracle":
        run_oracle(cfg, args.dataset)
    elif args.cmd == "sanity":
        run_corrupt_eval(cfg, args.dataset)
    elif args.cmd == "score":
        score_run(cfg, args.dataset, args.run_id)
    elif args.cmd == "restyle":
        restyle_run_html(cfg, args.dataset, args.run_id)
    elif args.cmd == "reconvert":
        reconvert_run(cfg, args.dataset, args.run_id)
    elif args.cmd == "dataset-viewer":
        path = write_dataset_viewer(cfg, args.dataset, portable=args.portable)
        print(path)
    elif args.cmd == "run-ollama":
        run_ollama(
            cfg,
            args.dataset,
            args.model,
            args.run_id,
            prompt_id=args.prompt_id,
            timeout_seconds=args.timeout_seconds,
            num_predict=args.num_predict,
            num_ctx=args.num_ctx,
            min_image_side=args.min_image_side,
            long_image_side=args.long_image_side,
            preprocess_profile=args.preprocess_profile,
        )
    elif args.cmd == "run-openai-vision":
        run_openai_vision(
            cfg,
            args.dataset,
            args.model,
            args.run_id,
            base_url=args.base_url,
            prompt_id=args.prompt_id,
            timeout_seconds=args.timeout_seconds,
            max_tokens=args.max_tokens,
            min_image_side=args.min_image_side,
            long_image_side=args.long_image_side,
            preprocess_profile=args.preprocess_profile,
            top_k=args.top_k,
        )
    elif args.cmd == "import-artifact":
        data_dir = Path(args.data_dir)
        output_dir = Path(cfg["output_dir"])
        if args.source_json:
            import_artifact_json(Path(args.source_json), args.dataset, data_dir, output_dir)
        elif args.source_html:
            import_artifact_html(Path(args.source_html), args.dataset, data_dir, output_dir)
        elif args.images_dir and args.gt_dir:
            import_paired_dirs(Path(args.images_dir), Path(args.gt_dir), args.dataset, data_dir, output_dir)
        else:
            raise SystemExit("Use either --source-html, or both --images-dir and --gt-dir.")


if __name__ == "__main__":
    main()
