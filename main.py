"""
Main runner: loads cached results for all levels and prints metrics.

Usage:
    - python main.py              # load saved results (no API calls)
    - python main.py --rerun      # re-run all levels via the API
    - python main.py --levels 2 3 # load/run specific levels
"""

import os
import sys
import json
import asyncio
import argparse
import subprocess
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "FActScore"))

from sklearn.metrics import classification_report, accuracy_score, f1_score
from level1 import run_level1
from level2 import run_level2, compute_metrics as l2_compute_metrics

ROOT        = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
LEVEL3_DIR  = ROOT / "level3"
LABELS      = ["SUPPORT", "CONTRADICT", "NEI"]


# ── helpers ────────────────────────────────────────────────────────────────────

def _banner(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def _report(label: str, gt: list, pred: list):
    print(f"\n{label}")
    print(classification_report(gt, pred, labels=LABELS, zero_division=0))


def _metrics(gt: list, pred: list) -> tuple[float, float]:
    acc = accuracy_score(gt, pred)
    mf1 = f1_score(gt, pred, labels=LABELS, average="macro", zero_division=0)
    return acc, mf1


def _load_json(path: Path) -> list | None:
    if path.exists():
        return json.loads(path.read_text())
    return None


# ── level display functions ───────────────────────────────────────────────────

def display_level1(data: list):
    gt      = [r["ground_truth"] for r in data]
    fs_pred = [r["fs_verdict"]   for r in data]
    uq_pred = [r["uqlm_verdict"] for r in data]
    _report("FActScore L1  vs Ground Truth", gt, fs_pred)
    _report("uqlm L1       vs Ground Truth", gt, uq_pred)
    agree = sum(1 for r in data if r["fs_verdict"] == r["uqlm_verdict"])
    print(f"FActScore ↔ uqlm agreement: {agree}/{len(data)} ({100*agree/len(data):.1f}%)")
    return gt, fs_pred, uq_pred


def display_level2(data: list):
    gt      = [r["ground_truth"] for r in data]
    fs_pred = [r["fs_verdict"]   for r in data]
    uq_pred = [r["uqlm_verdict"] for r in data]
    _report("FActScore L2  vs Ground Truth", gt, fs_pred)
    _report("uqlm L2       vs Ground Truth", gt, uq_pred)
    agree = sum(1 for r in data if r["fs_verdict"] == r["uqlm_verdict"])
    print(f"FActScore ↔ uqlm agreement: {agree}/{len(data)} ({100*agree/len(data):.1f}%)")
    return gt, fs_pred, uq_pred


def display_level3(data: list):
    gt       = [r["ground_truth"]    for r in data]
    fs_pred  = [r["fs_verdict"]      for r in data]
    uq_pred  = [r["uqlm_verdict"]    for r in data]
    nli_pred = [r.get("nli_verdict") for r in data]
    ens_pred = [r["ensemble_verdict"] for r in data]

    _report("FActScore L3  vs Ground Truth", gt, fs_pred)
    _report("uqlm L3       vs Ground Truth", gt, uq_pred)
    if any(v is not None for v in nli_pred):
        _report("NLI L3        vs Ground Truth", gt, nli_pred)
    _report("Ensemble L3   vs Ground Truth", gt, ens_pred)

    agree = sum(1 for r in data if r["fs_verdict"] == r["uqlm_verdict"])
    print(f"FActScore ↔ uqlm agreement: {agree}/{len(data)} ({100*agree/len(data):.1f}%)")
    methods = Counter(r.get("ensemble_method", "") for r in data)
    print(f"Ensemble methods: {dict(methods)}")

    disagree = [r for r in data if r["fs_verdict"] != r["uqlm_verdict"]]
    if disagree:
        print(f"\nDisagreements ({len(disagree)} cases):")
        print(f"  {'GT':<12} {'FActScore':<14} {'uqlm':<14} {'Ensemble':<13} Claim")
        print("  " + "-"*72)
        for r in disagree[:12]:
            fs_m  = "✓" if r["fs_verdict"]       == r["ground_truth"] else "✗"
            uq_m  = "✓" if r["uqlm_verdict"]     == r["ground_truth"] else "✗"
            en_m  = "✓" if r["ensemble_verdict"]  == r["ground_truth"] else "✗"
            print(f"  {r['ground_truth']:<12} {r['fs_verdict']}{fs_m:<13} "
                  f"{r['uqlm_verdict']}{uq_m:<13} {r['ensemble_verdict']}{en_m:<12} "
                  f"{r['claim'][:42]}")

    return gt, fs_pred, uq_pred, nli_pred, ens_pred


# ── cross-level summary ───────────────────────────────────────────────────────

def print_summary(rows: list[tuple[str, list, list]]):
    _banner("FULL RESULTS SUMMARY")
    print(f"\n  {'System':<26} {'Accuracy':>10}  {'Macro F1':>10}")
    print(f"  {'-'*50}")
    for label, gt, pred in rows:
        acc, mf1 = _metrics(gt, pred)
        print(f"  {label:<26} {acc*100:>9.1f}%  {mf1:>10.3f}")
    print()


# ── level runners ─────────────────────────────────────────────────────────────

async def level1_from_results(rerun: bool) -> tuple | None:
    if rerun:
        await run_level1(rerun=True)
    data = _load_json(RESULTS_DIR / "level1_results.json")
    if data is None:
        print("  [!] results/level1_results.json not found — skipping Level 1.")
        print("      Run with --rerun to generate results via the API.")
        return None
    print(f"  Loaded {len(data)} records from results/level1_results.json")
    return display_level1(data)


async def level2_from_results(rerun: bool) -> tuple | None:
    if rerun:
        metrics = await run_level2(rerun=True)
        data = _load_json(RESULTS_DIR / "level2_results.json")
    else:
        data = _load_json(RESULTS_DIR / "level2_results.json")
        if data is None:
            print("  [!] results/level2_results.json not found — skipping Level 2.")
            print("      Run with --rerun to generate results via the API.")
            return None
        print(f"  Loaded {len(data)} records from results/level2_results.json")
    return display_level2(data)


def level3_from_results(rerun: bool) -> tuple | None:
    if rerun:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "FActScore")
        env["PYTHONIOENCODING"] = "utf-8"
        rc = subprocess.run(
            [sys.executable, "level3.py"],
            cwd=str(LEVEL3_DIR), env=env,
        ).returncode
        if rc != 0:
            print(f"  [!] Level 3 exited with code {rc}")
            return None

    data = _load_json(RESULTS_DIR / "level3_results.json")
    if data is None:
        print("  [!] results/level3_results.json not found — skipping Level 3.")
        return None
    print(f"  Loaded {len(data)} records from results/level3_results.json")
    return display_level3(data)


# ── main ──────────────────────────────────────────────────────────────────────

async def main(levels: list[int], rerun: bool):
    summary_rows: list[tuple[str, list, list]] = []

    if 1 in levels:
        _banner("LEVEL 1 — FActScore + uqlm baseline")
        result = await level1_from_results(rerun)
        if result:
            gt, fs_pred, uq_pred = result
            summary_rows += [("L1  FActScore", gt, fs_pred), ("L1  uqlm", gt, uq_pred)]

    if 2 in levels:
        _banner("LEVEL 2 — Gated FActScore + Label-Prompted uqlm")
        result = await level2_from_results(rerun)
        if result:
            gt, fs_pred, uq_pred = result
            summary_rows += [("L2  FActScore", gt, fs_pred), ("L2  uqlm", gt, uq_pred)]

    if 3 in levels:
        _banner("LEVEL 3 — Ensemble: FActScore + uqlm + NLI")
        result = level3_from_results(rerun)
        if result:
            gt, fs_pred, uq_pred, nli_pred, ens_pred = result
            summary_rows += [
                ("L3  FActScore",  gt, fs_pred),
                ("L3  uqlm",       gt, uq_pred),
                ("L3  NLI",        gt, nli_pred) if any(v for v in nli_pred) else None,
                ("L3  Ensemble",   gt, ens_pred),
            ]
            summary_rows = [r for r in summary_rows if r is not None]

    if summary_rows:
        print_summary(summary_rows)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run / display all pipeline levels.")
    parser.add_argument(
        "--rerun", action="store_true",
        help="Re-run pipelines via the API instead of loading saved results.",
    )
    parser.add_argument(
        "--levels", nargs="+", type=int, default=[1, 2, 3],
        metavar="N", help="Which levels to run (default: 1 2 3).",
    )
    args = parser.parse_args()
    asyncio.run(main(levels=args.levels, rerun=args.rerun))