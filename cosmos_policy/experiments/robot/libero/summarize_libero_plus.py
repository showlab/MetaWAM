"""
Summarize sharded LIBERO-Plus eval logs into a per-perturbation-category success table, and optionally export
the task ids that still need to be (re-)evaluated.

The logs are the ENV_EVAL-*.txt files written by run_libero_eval.py when run with
`--task_num_shards N --task_shard_index i` and a `--run_id_note` that contains `--sh<i>of<N>--`.

Task ids are read from the "Task id: N" lines written by run_libero_eval.py. For older logs without these lines,
the ids are reconstructed from the sharding rule (shard i processes task_indices[i::num_shards] in ascending
order) and checked against the logged instruction; this fallback is only valid if the run did not use
--task_ids, so a warning is printed.

It also works on partially finished runs (only completed episodes are counted).

Usage (with the same LIBERO-Plus environment as for evaluation, i.e. LIBERO-Plus on PYTHONPATH):

    python -m cosmos_policy.experiments.robot.libero.summarize_libero_plus \
        --log_dir cosmos_policy/experiments/robot/libero/logs_libero_plus/ \
        --libero_plus_root <LIBERO_PLUS_ROOT> \
        --run_tag liberoplus--5stepAct--deterministic \
        --resume_out cosmos_policy/experiments/robot/libero/logs_libero_plus/task_ids/
"""

import argparse
import collections
import glob
import json
import os
import re
from typing import Dict, List, Optional, Tuple

SHARD_RE = re.compile(r"--sh(\d+)of(\d+)--")
SUITE_RE = re.compile(r"ENV_EVAL-(libero_[a-z0-9_]+)-cosmos-")


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize LIBERO-plus sharded eval logs by perturbation category")
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument(
        "--run_tag", type=str, default="", help="Only use logs whose file name contains this string (to select a run)"
    )
    parser.add_argument(
        "--libero_plus_root",
        type=str,
        default=os.environ.get("LIBERO_PLUS_ROOT", ""),
        help="LIBERO-Plus checkout (contains libero/libero/benchmark/task_classification.json); "
        "defaults to $LIBERO_PLUS_ROOT",
    )
    parser.add_argument("--json_out", type=str, default="", help="Optionally also write the results to this JSON file")
    parser.add_argument(
        "--resume_out", type=str, default="", help="Directory: export the unfinished task ids of each suite"
    )
    parser.add_argument(
        "--repair_out",
        type=str,
        default="",
        help=(
            "Directory: export the task ids that need to be re-evaluated. Unlike --resume_out, tasks whose log "
            "contains 'Episode error:' are included too (e.g. to redo CUDA OOM / renderer errors)."
        ),
    )
    parser.add_argument(
        "--repair_extra_log_dir",
        action="append",
        default=[],
        help=(
            "Extra log directory counted towards repair coverage (can be repeated). Only used with --repair_out: "
            "tasks that finished cleanly in these directories are removed from the next repair list."
        ),
    )
    args = parser.parse_args()
    if not args.libero_plus_root:
        parser.error("--libero_plus_root (or $LIBERO_PLUS_ROOT) is required")
    return args


def parse_log(path: str) -> List[Tuple[Optional[int], str, bool, bool]]:
    """Return (task_id or None, instruction, success, errored) for each completed episode, in order.

    The `except Exception` in run_episode swallows errors raised mid-rollout and ends the episode with
    success=False (only an `Episode error:` line is logged). Such errors-counted-as-failures silently lower the
    score, so they are flagged separately.
    """
    episodes: List[Tuple[Optional[int], str, bool, bool]] = []
    task_id: Optional[int] = None
    instruction: Optional[str] = None
    errored = False
    with open(path, errors="ignore") as file:
        for line in file:
            if line.startswith("Task id: "):
                task_id = int(line[len("Task id: ") :])
            elif line.startswith("Task: "):
                instruction = line[len("Task: ") :].strip()
            elif line.startswith("Episode error:"):
                errored = True
            elif line.startswith("Success: ") and instruction is not None:
                episodes.append((task_id, instruction, line.strip().endswith("True"), errored))
                instruction = None
                task_id = None
                errored = False
    return episodes


def main():
    args = parse_args()
    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    classification = json.load(
        open(os.path.join(args.libero_plus_root, "libero", "libero", "benchmark", "task_classification.json"))
    )

    paths = sorted(glob.glob(os.path.join(args.log_dir, "*.txt")))
    paths = [p for p in paths if args.run_tag in os.path.basename(p) and SHARD_RE.search(os.path.basename(p))]
    if not paths:
        raise SystemExit(f"No sharded logs matching {args.run_tag!r} under {args.log_dir}")

    stats: Dict[str, Dict[str, List[int]]] = collections.defaultdict(
        lambda: collections.defaultdict(lambda: [0, 0])
    )
    done_ids: Dict[str, set] = collections.defaultdict(set)
    clean_done_ids: Dict[str, set] = collections.defaultdict(set)
    errored_ids: Dict[str, set] = collections.defaultdict(set)
    attempts: Dict[Tuple[str, int], List[Tuple[str, bool, bool]]] = collections.defaultdict(list)
    suite_cache: Dict[str, tuple] = {}
    reconstructed = mismatches = 0

    for path in paths:
        name = os.path.basename(path)
        suite = SUITE_RE.search(name).group(1)
        shard_index, num_shards = (int(g) for g in SHARD_RE.search(name).groups())

        if suite not in suite_cache:
            task_suite = benchmark_dict[suite]()
            suite_cache[suite] = (task_suite, {row["id"] - 1: row["category"] for row in classification[suite]})
        task_suite, id_to_category = suite_cache[suite]
        fallback_ids = list(range(task_suite.n_tasks))[shard_index::num_shards]

        for k, (task_id, instruction, success, errored) in enumerate(parse_log(path)):
            if task_id is None:
                # Older logs: reconstruct the task id from the sharding rule and check it against the instruction
                if k >= len(fallback_ids):
                    raise SystemExit(f"{name}: more episodes than tasks in this shard; the log may have been reused")
                task_id = fallback_ids[k]
                reconstructed += 1
                if task_suite.get_task(task_id).language.strip() != instruction:
                    mismatches += 1
                    if mismatches <= 3:
                        print(f"[warn] {name} episode {k} does not match, skipping:\n  logged {instruction!r}")
                    continue
            attempts[(suite, task_id)].append((instruction, success, errored))

    if reconstructed:
        print(
            f"[warn] Task ids of {reconstructed} episodes were reconstructed (no 'Task id:' line in the log). "
            "This is only reliable if the run did not use --task_ids."
        )
    if mismatches:
        raise SystemExit(
            f"{mismatches} instructions do not match the reconstructed task ids; the categorization is unreliable, "
            "check the sharding arguments"
        )

    # A resumed repair run can contain an old errored Success=False record and a
    # later clean record for the same task.  Count each (suite, task_id) once,
    # preferring the latest clean attempt; otherwise retain the latest errored
    # attempt for an honest interim score and keep it in the repair set.
    total_episodes = errored_episodes = duplicate_attempts = conflicting_retries = 0
    for (suite, task_id), task_attempts in attempts.items():
        clean_attempts = [attempt for attempt in task_attempts if not attempt[2]]
        selected = clean_attempts[-1] if clean_attempts else task_attempts[-1]
        _, success, errored = selected
        _, id_to_category = suite_cache[suite]
        entry = stats[suite][id_to_category[task_id]]
        entry[0] += int(success)
        entry[1] += 1
        done_ids[suite].add(task_id)
        if any(attempt[2] for attempt in task_attempts):
            errored_ids[suite].add(task_id)
        if clean_attempts:
            clean_done_ids[suite].add(task_id)
        total_episodes += 1
        errored_episodes += int(errored)
        duplicate_attempts += len(task_attempts) - 1
        if len({attempt[1] for attempt in task_attempts}) > 1:
            conflicting_retries += 1

    categories = sorted({c for suite_stats in stats.values() for c in suite_stats})
    width = max(len(c) for c in categories) + 2
    suites = list(stats)

    print(f"\n{total_episodes} completed episodes")
    if duplicate_attempts:
        print(
            f"[resume] Deduplicated {duplicate_attempts} repeated attempts: each (suite, task_id) is counted once, "
            f"preferring clean re-runs; the result changed on retry for {conflicting_retries} tasks."
        )
    if errored_episodes:
        print(
            f"[warn] {errored_episodes} of them errored mid-rollout and were recorded as failures by run_episode "
            f"({errored_episodes / total_episodes * 100:.2f}%). grep 'Episode error:' for the causes."
        )
    print()
    header = f"{'category':<{width}}" + "".join(f"{s:>18}" for s in suites) + f"{'total':>14}"
    print(header)
    print("-" * len(header))

    grand = [0, 0]
    for category in categories:
        row = f"{category:<{width}}"
        cat_total = [0, 0]
        for suite in suites:
            ok, n = stats[suite].get(category, [0, 0])
            cat_total[0] += ok
            cat_total[1] += n
            row += f"{(f'{ok}/{n} ({ok / n * 100:.1f}%)' if n else '-'):>18}"
        grand[0] += cat_total[0]
        grand[1] += cat_total[1]
        row += f"{(f'{cat_total[0] / cat_total[1] * 100:.1f}%' if cat_total[1] else '-'):>14}"
        print(row)

    print("-" * len(header))
    row = f"{'TOTAL':<{width}}"
    for suite in suites:
        ok = sum(v[0] for v in stats[suite].values())
        n = sum(v[1] for v in stats[suite].values())
        row += f"{(f'{ok}/{n} ({ok / n * 100:.1f}%)' if n else '-'):>18}"
    row += f"{(f'{grand[0] / grand[1] * 100:.1f}%' if grand[1] else '-'):>14}"
    print(row)

    if args.json_out:
        with open(args.json_out, "w") as file:
            json.dump({s: dict(cs) for s, cs in stats.items()}, file, indent=1)
        print(f"\nwrote {args.json_out}")

    if args.resume_out:
        os.makedirs(args.resume_out, exist_ok=True)
        print()
        for suite in suites:
            task_suite, _ = suite_cache[suite]
            remaining = [i for i in range(task_suite.n_tasks) if i not in done_ids[suite]]
            path = os.path.join(args.resume_out, f"resume_{suite}.json")
            with open(path, "w") as file:
                json.dump(remaining, file)
            print(f"[resume] {suite}: {len(remaining)}/{task_suite.n_tasks} remaining -> {path}")

    if args.repair_out:
        # A repair round usually writes to its own directory. Tasks that finished cleanly there are treated as
        # covering the main logs, which yields the minimal list for the next repair round; they are not added to
        # the scores above to avoid double counting.
        for extra_dir in args.repair_extra_log_dir:
            extra_paths = sorted(glob.glob(os.path.join(extra_dir, "*.txt")))
            for path in extra_paths:
                name = os.path.basename(path)
                if not SHARD_RE.search(name):
                    continue
                suite_match = SUITE_RE.search(name)
                if suite_match is None:
                    continue
                suite = suite_match.group(1)
                if suite not in suite_cache:
                    task_suite = benchmark_dict[suite]()
                    suite_cache[suite] = (
                        task_suite,
                        {row["id"] - 1: row["category"] for row in classification[suite]},
                    )
                for task_id, _, _, errored in parse_log(path):
                    if task_id is None:
                        raise SystemExit(f"{name}: repair log has no 'Task id:' lines, cannot merge it safely")
                    if errored:
                        errored_ids[suite].add(task_id)
                    else:
                        clean_done_ids[suite].add(task_id)

        os.makedirs(args.repair_out, exist_ok=True)
        print()
        for suite in suites:
            task_suite, _ = suite_cache[suite]
            # "Done" is stricter for repair: only a final state without Episode error counts as done, so tasks that
            # never logged a Success line (e.g. FBO errors / killed processes) are included as well.
            repair = [i for i in range(task_suite.n_tasks) if i not in clean_done_ids[suite]]
            path = os.path.join(args.repair_out, f"repair_{suite}.json")
            with open(path, "w") as file:
                json.dump(repair, file)
            print(
                f"[repair] {suite}: re-run {len(repair)}/{task_suite.n_tasks} "
                f"(tasks with a logged Episode error: {len(errored_ids[suite])}) -> {path}"
            )


if __name__ == "__main__":
    main()
