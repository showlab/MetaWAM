from __future__ import annotations

import argparse
import json
import os
import pickle
from collections import OrderedDict
from pathlib import Path

import torch
from filelock import FileLock
from tqdm import tqdm

from cosmos_policy._src.predict2.inference.get_t5_emb import get_text_embedding


def _load_texts(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    if path.suffix.lower() == ".json":
        data = json.loads(raw)
        if isinstance(data, list):
            return [str(item["text"] if isinstance(item, dict) and "text" in item else item) for item in data]
        raise ValueError(f"Expected a JSON list in {path}")

    texts = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            if isinstance(record, dict) and "text" in record:
                texts.append(str(record["text"]))
            elif isinstance(record, str):
                texts.append(record)
            else:
                texts.append(line)
        except json.JSONDecodeError:
            texts.append(line)
    return texts


def _dedupe_preserve_order(texts: list[str]) -> list[str]:
    result = OrderedDict()
    for text in texts:
        text = text.strip()
        if text:
            result[text] = None
    return list(result.keys())


def _load_existing(output_path: Path) -> dict[str, torch.Tensor]:
    if not output_path.exists():
        return {}
    lock = FileLock(str(output_path) + ".lock", timeout=30)
    with lock:
        with output_path.open("rb") as file:
            return pickle.load(file)


def _save_embeddings(output_path: Path, embeddings: dict[str, torch.Tensor]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(output_path) + ".lock", timeout=30)
    with lock:
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        with tmp_path.open("wb") as file:
            pickle.dump(embeddings, file)
        os.replace(tmp_path, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute T5 embeddings from eval-collected text commands.")
    parser.add_argument("--texts_path", required=True, help="JSONL/TXT/JSON file containing text commands.")
    parser.add_argument("--output_path", required=True, help="Output pkl path for extra T5 embeddings.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore an existing output pkl instead of merging missing keys into it.",
    )
    args = parser.parse_args()

    texts_path = Path(args.texts_path)
    output_path = Path(args.output_path)
    texts = _dedupe_preserve_order(_load_texts(texts_path))
    embeddings = {} if args.overwrite else _load_existing(output_path)
    pending = [text for text in texts if text not in embeddings]

    print(f"Loaded {len(texts)} unique texts from {texts_path}")
    print(f"Existing embeddings: {len(embeddings)}; pending: {len(pending)}")
    for text in tqdm(pending, desc="Computing T5 embeddings"):
        embeddings[text] = get_text_embedding(text).to(dtype=torch.bfloat16).cpu()

    _save_embeddings(output_path, embeddings)
    print(f"Saved {len(embeddings)} T5 embeddings to {output_path}")


if __name__ == "__main__":
    main()
