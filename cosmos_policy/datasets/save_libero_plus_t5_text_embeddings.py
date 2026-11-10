"""
Precomputes T5 text embeddings for all LIBERO-plus task instructions.

LIBERO-plus has ~10k unique instructions, while the released `libero_t5_embeddings.pkl` covers only the 40 original
ones. Without precomputing them, evaluation computes an embedding on every cache miss and rewrites the whole pkl each
time, which is impractical for a full suite.

Compared to the per-instruction loop in `save_libero_t5_text_embeddings.py`, this script:
  1. **Uses dynamic padding.** Instructions are only ~10-30 tokens long; the reference implementation pads each one
     to 512 tokens before the forward pass. Here each batch is encoded at its true length and zero-padded to
     (1, 512, 1024) afterwards. The T5 encoder uses bidirectional attention with an attention mask, so masked
     positions do not affect the real tokens and the result matches the reference implementation (check with
     --verify).
  2. **Buckets by length.** Instructions are sorted by token length and batched under a token budget, which
     minimizes padding within a batch.
  3. **Runs data-parallel across GPUs.** The instruction list is sharded, each GPU process loads its own encoder,
     and the main process merges the results. Loading the T5-11B encoder (~19GB of fp32 weights) is often slower
     than inference itself, so extra GPUs mainly speed up the inference part.

The output format matches the released pkl: dict[str, torch.Tensor] with CPU bfloat16 values of shape (1, 512, 1024).
Keys are strip()-ed, consistent with `_normalize_t5_text` on the evaluation side.

Usage (the LIBERO-plus environment must be importable, since the script imports `libero` to list the tasks):

    PYTHONPATH=<LIBERO_PLUS_ROOT>:<LIBERO_PLUS_DEPS> \
    MAGICK_HOME=<IMAGEMAGICK_ROOT> \
    LD_LIBRARY_PATH=<IMAGEMAGICK_ROOT>/lib:$LD_LIBRARY_PATH \
    LIBERO_CONFIG_PATH=$HOME/.libero_plus \
    HF_HOME=<HF_CACHE_DIR> \
    python -m cosmos_policy.datasets.save_libero_plus_t5_text_embeddings \
        --out_dir <OUTPUT_DIR>/liberoplus_t5 \
        --gpus 0,1,2,3 \
        --merge_pkl nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl

Then point --t5_text_embeddings_path at <out_dir>/libero_plus_t5_embeddings__<suite>.pkl during evaluation.
(The per-suite split is intentional: evaluation moves the whole cache to the GPU, ~2.6GB per suite versus ~10.5GB for
the merged file, and every worker holds its own copy.)
"""

import argparse
import os
import pickle
import time
from typing import Dict, List, Sequence

import torch
import torch.multiprocessing as mp

DEFAULT_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
NUM_TOKENS = 512
EMBED_DIM = 1024


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute T5 text embeddings for all LIBERO-plus instructions (multi-GPU, batched)"
    )
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated GPU ids for data-parallel sharding")
    parser.add_argument(
        "--suites", type=str, default=",".join(DEFAULT_SUITES), help="Comma-separated task suites"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="both",
        choices=["suite", "all", "both"],
        help="suite: one pkl per suite; all: a single merged pkl; both: write both",
    )
    parser.add_argument(
        "--merge_pkl",
        type=str,
        action="append",
        default=[],
        help="Merge entries from an existing pkl into the result (repeatable). HF paths are supported; existing keys are not recomputed",
    )
    parser.add_argument(
        "--instructions_json",
        type=str,
        default="",
        help="Read {suite: [instruction, ...]} from a json file instead of importing libero",
    )
    parser.add_argument(
        "--dump_instructions", type=str, default="", help="Only dump the instruction list to this json file and exit"
    )
    parser.add_argument(
        "--max_tokens_per_batch",
        type=int,
        default=32768,
        help="Token budget per batch (the batch size adapts to the current sequence length)",
    )
    parser.add_argument("--max_batch_size", type=int, default=512, help="Maximum number of instructions per batch")
    parser.add_argument("--bf16", action="store_true", help="Run the encoder in bf16 (faster, with small numerical differences from the reference implementation)")
    parser.add_argument("--tf32", action="store_true", help="Allow TF32 matmuls (faster, with small numerical differences)")
    parser.add_argument(
        "--verify", type=int, default=8, help="Compare N sampled instructions against the reference implementation (padded to 512); 0 disables the check"
    )
    parser.add_argument("--model_name", type=str, default="google-t5/t5-11b")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    return parser.parse_args()


# --------------------------------------------------------------------------------------
# Instruction collection
# --------------------------------------------------------------------------------------
def collect_instructions(suites: Sequence[str]) -> Dict[str, List[str]]:
    """Collect all instructions of each suite from the LIBERO(-plus) benchmark."""
    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    out: Dict[str, List[str]] = {}
    for suite in suites:
        if suite not in benchmark_dict:
            raise KeyError(f"Unknown task suite {suite!r}. Available: {sorted(benchmark_dict)}")
        task_suite = benchmark_dict[suite]()
        seen = {}  # Order-preserving de-duplication
        for task_id in range(task_suite.n_tasks):
            seen.setdefault(task_suite.get_task(task_id).language.strip(), None)
        out[suite] = list(seen)
        print(f"[collect] {suite}: {task_suite.n_tasks} tasks -> {len(out[suite])} unique instructions")
    return out


def load_existing(paths: Sequence[str]) -> Dict[str, torch.Tensor]:
    """Load an existing pkl (HF paths supported) so its entries are not recomputed."""
    merged: Dict[str, torch.Tensor] = {}
    for path in paths:
        local_path = path
        if not os.path.exists(local_path):
            from cosmos_policy.experiments.robot.cosmos_utils import resolve_path

            local_path = resolve_path(path)
        with open(local_path, "rb") as file:
            data = pickle.load(file)
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                merged[str(key).strip()] = value.to(dtype=torch.bfloat16).cpu()
        print(f"[merge] loaded {len(data)} embeddings from {local_path}")
    return merged


# --------------------------------------------------------------------------------------
# Single-GPU worker
# --------------------------------------------------------------------------------------
def _make_batches(lengths: Sequence[int], max_tokens: int, max_bs: int) -> List[List[int]]:
    """Sort by token length and split into batches under a token budget; returns index groups."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches, cur = [], []
    for idx in order:
        cand = cur + [idx]
        # Pad to the longest sequence in the batch (the last one, since the list is sorted)
        if len(cand) > max_bs or len(cand) * lengths[idx] > max_tokens:
            if cur:
                batches.append(cur)
            cur = [idx]
        else:
            cur = cand
    if cur:
        batches.append(cur)
    return batches


def encode_shard(
    texts: List[str],
    gpu_id: int,
    model_name: str,
    max_tokens_per_batch: int,
    max_batch_size: int,
    use_bf16: bool,
    rank: int,
) -> Dict[str, torch.Tensor]:
    """Encode a list of instructions on the given GPU into (1, 512, 1024) bf16 CPU tensors."""
    from cosmos_policy._src.predict2.inference.get_t5_emb import CosmosT5TextEncoder

    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    t0 = time.time()
    encoder = CosmosT5TextEncoder(model_name=model_name, device=device)
    if use_bf16:
        encoder.text_encoder = encoder.text_encoder.to(dtype=torch.bfloat16)
    print(f"[gpu{gpu_id}] encoder ready in {time.time() - t0:.1f}s, {len(texts)} instructions", flush=True)

    tokenizer = encoder.tokenizer
    encoded = tokenizer(texts, padding=False, truncation=True, max_length=NUM_TOKENS)
    lengths = [len(ids) for ids in encoded["input_ids"]]
    batches = _make_batches(lengths, max_tokens_per_batch, max_batch_size)

    result: Dict[str, torch.Tensor] = {}
    t0 = time.time()
    with torch.inference_mode():
        for batch_idx, index_group in enumerate(batches):
            padded = tokenizer.pad(
                {"input_ids": [encoded["input_ids"][i] for i in index_group]},
                padding="longest",
                return_tensors="pt",
            )
            input_ids = padded["input_ids"].to(device)
            attn_mask = padded["attention_mask"].to(device)

            hidden = encoder.text_encoder(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state
            hidden = hidden.to(dtype=torch.bfloat16).cpu()

            for row, i in enumerate(index_group):
                emb = torch.zeros(1, NUM_TOKENS, EMBED_DIM, dtype=torch.bfloat16)
                length = lengths[i]
                # The reference implementation also zeros positions outside the attention mask, so this is equivalent
                emb[0, :length] = hidden[row, :length]
                result[texts[i]] = emb

            if batch_idx % 20 == 0 or batch_idx == len(batches) - 1:
                done = sum(len(g) for g in batches[: batch_idx + 1])
                print(
                    f"[gpu{gpu_id}] batch {batch_idx + 1}/{len(batches)} "
                    f"({done}/{len(texts)}) {time.time() - t0:.1f}s",
                    flush=True,
                )

    print(f"[gpu{gpu_id}] done, {len(result)} embeddings in {time.time() - t0:.1f}s", flush=True)
    return result


def _worker(rank: int, gpu_ids: List[int], shards: List[List[str]], args, tmp_dir: str):
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    embeddings = encode_shard(
        texts=shards[rank],
        gpu_id=gpu_ids[rank],
        model_name=args.model_name,
        max_tokens_per_batch=args.max_tokens_per_batch,
        max_batch_size=args.max_batch_size,
        use_bf16=args.bf16,
        rank=rank,
    )
    # Exchange results through a file rather than a pipe: a shard can be several GB
    with open(os.path.join(tmp_dir, f"shard_{rank}.pkl"), "wb") as file:
        pickle.dump(embeddings, file)


# --------------------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------------------
def verify_against_reference(
    embeddings: Dict[str, torch.Tensor], texts: Sequence[str], gpu_id: int, model_name: str, num: int
) -> None:
    """Compare samples against the reference implementation (padded to 512) to confirm dynamic padding is exact."""
    import random

    from cosmos_policy._src.predict2.inference.get_t5_emb import CosmosT5TextEncoder

    sample = random.sample(list(texts), min(num, len(texts)))
    encoder = CosmosT5TextEncoder(model_name=model_name, device=f"cuda:{gpu_id}")
    worst_ulp = 0.0
    with torch.inference_mode():
        for text in sample:
            ref = encoder.encode_prompts(text, max_length=NUM_TOKENS).to(dtype=torch.bfloat16).cpu().float()
            got = embeddings[text].float()
            diff = (ref - got).abs().max().item()
            scale = ref.abs().max().item()
            # bfloat16 has an 8-bit mantissa: 1 ULP at this magnitude
            ulp = 2.0 ** (int(torch.tensor(scale).log2().floor().item()) - 8) if scale > 0 else 1.0
            worst_ulp = max(worst_ulp, diff / ulp)
            mismatch = (ref != got).float().mean().item()
            print(
                f"[verify] max|diff|={diff:.3e} ({diff / ulp:.2f} bf16 ULP), "
                f"mismatched elements={mismatch:.2%}  <- {text[:55]!r}"
            )
    print(f"[verify] {len(sample)} samples, worst {worst_ulp:.2f} bf16 ULP")
    if worst_ulp <= 2.0:
        print(
            "[verify] OK: dynamic padding matches the reference within ~1e-6 relative error in fp32 (different matmul reduction order), "
            "i.e. at most 1 ULP after bf16 storage, which is expected"
        )
    else:
        print("[verify] Differences exceed 2 ULP; investigate (unless --bf16/--tf32 was used)")


# --------------------------------------------------------------------------------------
def main():
    args = parse_args()
    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    gpu_ids = [int(g) for g in args.gpus.split(",") if g.strip() != ""]
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Instruction list
    if args.instructions_json:
        import json

        with open(args.instructions_json, "r", encoding="utf-8") as file:
            per_suite = {k: [t.strip() for t in v] for k, v in json.load(file).items()}
        per_suite = {k: v for k, v in per_suite.items() if k in suites}
    else:
        per_suite = collect_instructions(suites)

    if args.dump_instructions:
        import json

        with open(args.dump_instructions, "w", encoding="utf-8") as file:
            json.dump(per_suite, file, ensure_ascii=False, indent=1)
        print(f"[collect] wrote instructions to {args.dump_instructions}")
        return

    all_texts = list(dict.fromkeys(t for texts in per_suite.values() for t in texts))
    print(f"[collect] {len(all_texts)} unique instructions across {len(per_suite)} suites")

    # 2) Reuse existing embeddings
    existing = load_existing(args.merge_pkl) if args.merge_pkl else {}
    todo = [t for t in all_texts if t not in existing]
    print(f"[plan] {len(existing)} reused, {len(todo)} to compute on GPUs {gpu_ids}")

    # 3) Multi-GPU data parallelism; round-robin sharding keeps length distributions balanced
    embeddings: Dict[str, torch.Tensor] = dict(existing)
    if todo:
        num_workers = min(len(gpu_ids), len(todo))
        shards = [todo[i::num_workers] for i in range(num_workers)]
        for rank, shard in enumerate(shards):
            print(f"[plan] gpu{gpu_ids[rank]} <- {len(shard)} instructions")

        tmp_dir = os.path.join(args.out_dir, ".shards")
        os.makedirs(tmp_dir, exist_ok=True)
        t0 = time.time()
        if num_workers == 1:
            _worker(0, gpu_ids, shards, args, tmp_dir)
        else:
            mp.spawn(_worker, args=(gpu_ids, shards, args, tmp_dir), nprocs=num_workers, join=True)
        print(f"[encode] all workers finished in {time.time() - t0:.1f}s")

        for rank in range(num_workers):
            shard_path = os.path.join(tmp_dir, f"shard_{rank}.pkl")
            with open(shard_path, "rb") as file:
                embeddings.update(pickle.load(file))
            os.remove(shard_path)
        os.rmdir(tmp_dir)

    missing = [t for t in all_texts if t not in embeddings]
    if missing:
        raise RuntimeError(f"{len(missing)} instructions missing after encoding, e.g. {missing[:3]}")

    # 4) Write to disk
    def dump(path: str, data: Dict[str, torch.Tensor]) -> None:
        if os.path.exists(path) and not args.overwrite:
            raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")
        with open(path, "wb") as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[save] {len(data):6d} embeddings -> {path} ({os.path.getsize(path) / 1e9:.2f} GB)")

    if args.split in ("suite", "both"):
        for suite, texts in per_suite.items():
            dump(
                os.path.join(args.out_dir, f"libero_plus_t5_embeddings__{suite}.pkl"),
                {t: embeddings[t] for t in texts},
            )
    if args.split in ("all", "both"):
        dump(os.path.join(args.out_dir, "libero_plus_t5_embeddings__all.pkl"), embeddings)

    # 5) Spot-check against the reference implementation
    if args.verify > 0:
        verify_against_reference(embeddings, todo or all_texts, gpu_ids[0], args.model_name, args.verify)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
