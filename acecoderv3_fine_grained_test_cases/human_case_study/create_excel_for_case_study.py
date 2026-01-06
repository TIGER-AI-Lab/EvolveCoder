#!/usr/bin/env python3
"""
Create an Excel workbook where each sheet corresponds to one QUESTION ID.
For each question ID:
  - verify the ID exists in EVERY provided Hugging Face dataset repo (else raise)
  - display the QUESTION once (from any repo; assumed identical across rounds)
  - display TEST CASES grouped by repo (round)

Requirements:
  pip install datasets openpyxl

Usage:
  1) Fill HF_URLS and QUESTION_IDS below
  2) python hf_to_excel_by_id.py
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from datasets import load_dataset
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


# ----------------------------
# CONFIG
# ----------------------------
HF_URLS = [
    "chiruan/acecoderv3_round0",
    "chiruan/acecoderv3_round1",
    "chiruan/acecoderv3_round2",
    "chiruan/acecoderv3_round3",
]

# Provide the question IDs you want to export (must exist in every repo)
QUESTION_IDS = [
    "TACO_4591",
    "primeintellect_211",
    "APPS_527",
    "TACO_10981",
    "contests_2890",
    "primeintellect_398",
    "APPS_1112",
    "contests_989",
    "TACO_1721",
    "codeforces_1069",
]

OUTPUT_XLSX = "human_study_by_id.xlsx"
SPLIT_PREFERENCE = ["train", "validation", "test"]  # first available will be used

# Excel cell hard limit is 32,767 characters; truncate to avoid write errors.
EXCEL_CELL_CHAR_LIMIT = 32767
TRUNCATE_QUESTION_TO = 30000

# Layout
SPACER_ROWS_BETWEEN_REPOS = 1


# ----------------------------
# HELPERS
# ----------------------------
def pick_split_name(ds_dict) -> str:
    keys = list(ds_dict.keys())
    for s in SPLIT_PREFERENCE:
        if s in ds_dict:
            return s
    return keys[0]


def load_one_dataset(url: str):
    ds = load_dataset(url)
    if hasattr(ds, "keys"):  # DatasetDict
        split = pick_split_name(ds)
        return ds[split], split
    return ds, "all"


def safe_sheet_name(name: str, existing: set[str]) -> str:
    """
    Excel sheet name rules:
      - max 31 chars
      - cannot contain: : \ / ? * [ ]
      - must be unique within workbook
    """
    name = re.sub(r"[:\\/?*\[\]]", "_", name).strip()
    if not name:
        name = "sheet"
    base = name[:31]

    if base not in existing:
        existing.add(base)
        return base

    i = 2
    while True:
        suffix = f"_{i}"
        trimmed = base[: 31 - len(suffix)]
        candidate = trimmed + suffix
        if candidate not in existing:
            existing.add(candidate)
            return candidate
        i += 1


def truncate_for_excel(s: str, limit: int) -> str:
    if s is None:
        return ""
    s = str(s)
    if len(s) <= limit:
        return s
    return s[:limit] + "\n\n[TRUNCATED]"


def set_col_width(ws, col: int, width: float):
    ws.column_dimensions[get_column_letter(col)].width = width


def style_cell(cell, wrap=True, valign="top", bold=False, font: Font | None = None):
    cell.alignment = Alignment(wrap_text=wrap, vertical=valign)
    if bold:
        cell.font = Font(bold=True) if font is None else Font(**{**font.copy().__dict__, "bold": True})
    elif font is not None:
        cell.font = font


def parse_round_label(url: str) -> str:
    """
    Best-effort label like 'round0' from repo name.
    e.g. 'chiruan/acecoderv3_round1' -> 'round1'
    """
    m = re.search(r"(round\d+)", url, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower()
    # fallback to last path component
    return url.split("/")[-1]


def build_id_index(ds) -> Dict[str, int]:
    """
    Build a mapping: id -> row index in the dataset.
    Assumes 'id' is unique within a split.
    """
    idx: Dict[str, int] = {}
    # Iterating through ds can be slow but is simplest and robust.
    for i, ex in enumerate(ds):
        ex_id = ex.get("id", None)
        if ex_id is None:
            continue
        # keep first occurrence if duplicates exist
        if ex_id not in idx:
            idx[ex_id] = i
    return idx


# ----------------------------
# MAIN
# ----------------------------
def main():
    if not HF_URLS:
        raise ValueError("HF_URLS is empty. Add your Hugging Face dataset IDs to HF_URLS.")
    if not QUESTION_IDS:
        raise ValueError("QUESTION_IDS is empty. Add at least one question ID.")

    # Load all datasets once, build id->row index for each
    datasets_info: List[Tuple[str, str, object, Dict[str, int]]] = []
    for url in HF_URLS:
        ds, split = load_one_dataset(url)

        required_cols = {"id", "question", "tests"}
        missing = required_cols - set(ds.column_names)
        if missing:
            raise ValueError(
                f"Dataset '{url}' (split '{split}') missing columns: {sorted(missing)}. "
                f"Found columns: {ds.column_names}"
            )

        id_to_row = build_id_index(ds)
        datasets_info.append((url, split, ds, id_to_row))

    # Validate every requested ID exists in every repo (fail fast)
    for qid in QUESTION_IDS:
        for (url, split, _ds, id_to_row) in datasets_info:
            if qid not in id_to_row:
                raise ValueError(
                    f"ID '{qid}' not found in dataset '{url}' (split '{split}'). "
                    f"Please check the ID list or the dataset split."
                )

    # Create workbook
    wb = Workbook()
    wb.remove(wb.active)
    used_sheet_names: set[str] = set()

    # Styles
    label_fill = PatternFill("solid", fgColor="F2F2F2")
    header_fill = PatternFill("solid", fgColor="D9E1F2")
    mono_font = Font(name="Consolas")
    bold_font = Font(bold=True)

    # For each question id, create one sheet
    for qid in QUESTION_IDS:
        sheet_name = safe_sheet_name(qid, used_sheet_names)
        ws = wb.create_sheet(title=sheet_name)

        # Two columns: A=Label, B=Content
        set_col_width(ws, 1, 18)
        set_col_width(ws, 2, 140)

        # Top info
        ws["A1"] = "Question ID"
        ws["B1"] = truncate_for_excel(qid, EXCEL_CELL_CHAR_LIMIT)
        ws["A2"] = "Instructions"
        ws["B2"] = "Top-to-bottom: QUESTION, then TESTS grouped by repo/round."
        for r in (1, 2):
            a = ws[f"A{r}"]
            b = ws[f"B{r}"]
            a.fill = label_fill
            a.font = bold_font
            style_cell(a, wrap=True, valign="top")
            style_cell(b, wrap=True, valign="top")

        ws.freeze_panes = "A3"
        cur_row = 4

        # Pull the question from the first repo (assumed identical across rounds)
        first_url, first_split, first_ds, first_map = datasets_info[0]
        first_ex = first_ds[int(first_map[qid])]
        question = truncate_for_excel(first_ex.get("question", ""), TRUNCATE_QUESTION_TO)

        # QUESTION block
        ws.cell(row=cur_row, column=1, value="QUESTION").fill = label_fill
        ws.cell(row=cur_row, column=1).font = bold_font
        q_cell = ws.cell(row=cur_row, column=2, value=question)
        style_cell(ws.cell(row=cur_row, column=1), wrap=True, valign="top")
        style_cell(q_cell, wrap=True, valign="top")
        ws.row_dimensions[cur_row].height = 260
        cur_row += 2  # one blank row after question

        # Tests grouped by repo
        for (url, split, ds, id_to_row) in datasets_info:
            round_label = parse_round_label(url)
            ex = ds[int(id_to_row[qid])]

            tests = ex.get("tests", []) or []
            tests = [truncate_for_excel(t, EXCEL_CELL_CHAR_LIMIT) for t in tests]

            # Repo header row
            ws.cell(row=cur_row, column=1, value="REPO").fill = header_fill
            ws.cell(row=cur_row, column=1).font = bold_font
            ws.cell(row=cur_row, column=2, value=f"{url}  (split: {split})  [{round_label}]")
            style_cell(ws.cell(row=cur_row, column=1), wrap=True, valign="top")
            style_cell(ws.cell(row=cur_row, column=2), wrap=True, valign="top")
            cur_row += 1

            # TESTS header row
            ws.cell(row=cur_row, column=1, value="TESTS").fill = label_fill
            ws.cell(row=cur_row, column=1).font = bold_font
            ws.cell(row=cur_row, column=2, value=f"{len(tests)} test cases (one per row below)")
            style_cell(ws.cell(row=cur_row, column=1), wrap=True, valign="top")
            style_cell(ws.cell(row=cur_row, column=2), wrap=True, valign="top")
            cur_row += 1

            # Each test per row
            if not tests:
                ws.cell(row=cur_row, column=1, value="test_1")
                c2 = ws.cell(row=cur_row, column=2, value="")
                style_cell(ws.cell(row=cur_row, column=1), wrap=False, valign="top")
                c2.font = mono_font
                style_cell(c2, wrap=True, valign="top")
                ws.row_dimensions[cur_row].height = 50
                cur_row += 1
            else:
                for j, t in enumerate(tests, start=1):
                    ws.cell(row=cur_row, column=1, value=f"test_{j}")
                    c2 = ws.cell(row=cur_row, column=2, value=t)
                    style_cell(ws.cell(row=cur_row, column=1), wrap=False, valign="top")
                    c2.font = mono_font
                    style_cell(c2, wrap=True, valign="top")
                    ws.row_dimensions[cur_row].height = 60
                    cur_row += 1

            # Spacer between repos
            cur_row += SPACER_ROWS_BETWEEN_REPOS

    wb.save(OUTPUT_XLSX)
    print(f"Wrote Excel file: {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
