# scripts/ingest_nerd_csv_semantics.py
"""
Ingest NEI-style CSV dumps (block.csv, item.csv) into generated semantics configs.

Inputs (from config/raw/):
  - item.csv
  - block.csv

Actual headers (from your dumps):

  item.csv:
    Name,ID,Has Block,Mod,Class,Display Name

  block.csv:
    Name,ID,Has Item,Mod,Class,Display Name

We only care about:
  - Name          -> registry id ("minecraft:stone")
  - Display Name  -> localized display name ("Stone")

Outputs (to config/):
  - gtnh_items.generated.yaml
  - gtnh_blocks.generated.yaml

Format:

  items:
    "minecraft:stone":
      default_category: "stone" | "unknown" | ...
      display_name: "Stone"

  blocks:
    "minecraft:stone":
      default_category: "stone" | "unknown" | ...
      display_name: "Stone"

Hand-authored gtnh_items.yaml / gtnh_blocks.yaml still override these via SemanticsDB.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Any, Optional

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "config" / "raw"
OUT_DIR = PROJECT_ROOT / "config"


# ---------------------------------------------------------------------------
# Heuristic category guessing (low-stakes, you override in hand-authored YAML)
# ---------------------------------------------------------------------------

def _infer_default_category(registry_id: str, display_name: Optional[str]) -> str:
    text = (registry_id + " " + (display_name or "")).lower()

    # Item-ish
    if "ingot" in text:
        return "ingot"
    if "dust" in text:
        return "dust"
    if "plate" in text and "chestplate" not in text:
        return "plate"
    if "nugget" in text:
        return "nugget"

    # Block-ish
    if "ore" in text:
        return "ore"
    if "log" in text:
        return "log"
    if "planks" in text or "plank" in text:
        return "plank"
    if "stone" in text:
        return "stone"
    if "dirt" in text:
        return "dirt"
    if "gravel" in text:
        return "gravel"
    if "sand" in text:
        return "sand"
    if "machine" in text or "generator" in text:
        return "machine"

    return "unknown"


# ---------------------------------------------------------------------------
# Generic CSV reader for your NEI dumps
# ---------------------------------------------------------------------------

def _read_csv_with_headers(path: Path) -> list[Dict[str, str]]:
    """
    Read a NEI dump CSV with comma delimiter and normalized headers.

    Assumes header row exists and matches:
      ["Name", "ID", "Has Block"/"Has Item", "Mod", "Class", "Display Name"]
    """
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    rows: list[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=",")
        # Normalize header keys by stripping whitespace once
        fieldnames = reader.fieldnames or []
        normalized = [h.strip() for h in fieldnames]
        # Map original headers -> stripped headers
        header_map = {orig: norm for orig, norm in zip(fieldnames, normalized)}

        for raw_row in reader:
            if not raw_row:
                continue
            row: Dict[str, str] = {}
            for orig_key, value in raw_row.items():
                norm_key = header_map.get(orig_key, orig_key).strip()
                row[norm_key] = value
            # Skip fully empty logical rows
            if not any(isinstance(v, str) and v.strip() for v in row.values()):
                continue
            rows.append(row)

    print(f"[ingest] {path.name}: parsed {len(rows)} rows")
    return rows


# ---------------------------------------------------------------------------
# Items ingestion (item.csv)
# ---------------------------------------------------------------------------

def ingest_items(item_csv: Path) -> Dict[str, Any]:
    """
    Convert item.csv into gtnh_items.generated.yaml structure:

      items:
        "mod:item":
          default_category: "..."
          display_name: "..."
    """
    rows = _read_csv_with_headers(item_csv)
    items: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        registry_id = (row.get("Name") or "").strip()
        if not registry_id:
            continue

        display_raw = (row.get("Display Name") or "").strip()
        # NEI literally writes "null" as text sometimes, we treat that as missing
        display_name = display_raw if display_raw and display_raw.lower() != "null" else None

        default_category = _infer_default_category(registry_id, display_name)

        if registry_id not in items:
            entry: Dict[str, Any] = {"default_category": default_category}
            if display_name:
                entry["display_name"] = display_name
            items[registry_id] = entry

    print(f"[ingest] Built {len(items)} item entries")
    return {"items": items}


# ---------------------------------------------------------------------------
# Blocks ingestion (block.csv)
# ---------------------------------------------------------------------------

def ingest_blocks(block_csv: Path) -> Dict[str, Any]:
    """
    Convert block.csv into gtnh_blocks.generated.yaml structure:

      blocks:
        "mod:block":
          default_category: "..."
          display_name: "..."
    """
    rows = _read_csv_with_headers(block_csv)
    blocks: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        registry_id = (row.get("Name") or "").strip()
        if not registry_id:
            continue

        display_raw = (row.get("Display Name") or "").strip()
        display_name = display_raw if display_raw and display_raw.lower() != "null" else None

        default_category = _infer_default_category(registry_id, display_name)

        if registry_id not in blocks:
            entry: Dict[str, Any] = {"default_category": default_category}
            if display_name:
                entry["display_name"] = display_name
            blocks[registry_id] = entry

    print(f"[ingest] Built {len(blocks)} block entries")
    return {"blocks": blocks}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    item_csv = RAW_DIR / "item.csv"
    block_csv = RAW_DIR / "block.csv"

    items_out = OUT_DIR / "gtnh_items.generated.yaml"
    blocks_out = OUT_DIR / "gtnh_blocks.generated.yaml"

    print(f"[ingest] RAW_DIR={RAW_DIR}")

    # Items
    print(f"[ingest] Reading items from {item_csv}")
    items_data = ingest_items(item_csv)
    with items_out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(items_data, f, sort_keys=True, allow_unicode=True)
    print(f"[ingest] Wrote {items_out}")

    # Blocks
    print(f"[ingest] Reading blocks from {block_csv}")
    blocks_data = ingest_blocks(block_csv)
    with blocks_out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(blocks_data, f, sort_keys=True, allow_unicode=True)
    print(f"[ingest] Wrote {blocks_out}")

    print("[ingest] Done. Hand-authored gtnh_items.yaml / gtnh_blocks.yaml still override these.")


if __name__ == "__main__":
    main()

