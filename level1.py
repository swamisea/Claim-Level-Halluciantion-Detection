"""
Level 1: FActScore + uqlm baseline on SciFact (150 balanced claims).

This script is designed to be called from a main runner script:

    from level1_scifact import run_level1

    # Default: load previously saved results (no API calls)
    metrics = asyncio.run(run_level1())

    # Force a fresh run (requires OPENAI_API_KEY)
    metrics = asyncio.run(run_level1(rerun=True))

CLI usage:
    python level1_scifact.py           # load saved results
    python level1_scifact.py --rerun   # re-run full pipeline

Returns a metrics dict and writes two files:
  results/level1_results.json   — per-sample predictions and raw outputs
  results/level1_metrics.json   — accuracy, macro_f1, and per-class recall/f1
"""

import os
import sys
import json
import asyncio
import random
import sqlite3
import argparse
from collections import Counter
from pathlib import Path

sys.path.insert(0, "FActScore")

from sklearn.metrics import accuracy_score, classification_report
from langchain_openai import ChatOpenAI
from uqlm import LongTextUQ
from factscore.factscorer import FactScorer
from factscore.retrieval import DocDB, Retrieval

# ── Config ─────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
DATA_DIR       = Path("data/scifact/data")
RESULTS_DIR    = Path("results")
SCIFACT_DB     = Path("data/scifact/scifact_corpus.db")
SCIFACT_CACHE  = Path("data/scifact/factscore_cache")
FACTSCORE_DATA = os.path.expanduser("~/.cache/factscore")

RESULTS_DIR.mkdir(exist_ok=True)
SCIFACT_CACHE.mkdir(parents=True, exist_ok=True)

N_CLAIMS  = 10
RAND_SEED = 42

SUPPORT    = "SUPPORT"
CONTRADICT = "CONTRADICT"
NEI        = "NEI"

SPECIAL_SEPARATOR = "####SPECIAL####SEPARATOR####"

FS_SUPPORT_THRESHOLD    = 0.6
FS_CONTRADICT_THRESHOLD = 0.4

LABEL_SCAN_CHARS = 60


# ── 1. Load corpus ─────────────────────────────────────────────────────────────

def load_corpus() -> dict:
    corpus = {}
    for line in (DATA_DIR / "corpus.jsonl").open():
        doc = json.loads(line)
        sentences = doc["abstract"]
        corpus[doc["doc_id"]] = {
            "title":    doc["title"],
            "abstract": sentences,
            "text":     SPECIAL_SEPARATOR.join(sentences),
        }
    return corpus


# ── 2. Build FActScore-compatible SQLite DB ────────────────────────────────────

def build_scifact_db(corpus: dict) -> str:
    db_path = str(SCIFACT_DB)
    if SCIFACT_DB.exists():
        return db_path

    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE documents (title PRIMARY KEY, text)")
    rows = [(doc["title"], doc["text"]) for doc in corpus.values()]
    conn.executemany("INSERT OR REPLACE INTO documents VALUES (?,?)", rows)
    conn.commit()
    conn.close()
    return db_path


# ── 3. Load claims ─────────────────────────────────────────────────────────────

def get_ground_truth(evidence: dict) -> str:
    if not evidence:
        return NEI
    labels = set()
    for v in evidence.values():
        for item in (v if isinstance(v, list) else [v]):
            if isinstance(item, dict):
                labels.add(item.get("label", ""))
    if SUPPORT    in labels: return SUPPORT
    if CONTRADICT in labels: return CONTRADICT
    return NEI


def load_claims(n: int = N_CLAIMS, seed: int = RAND_SEED) -> list:
    raw = [json.loads(l) for l in (DATA_DIR / "claims_train.jsonl").open()]

    by_label = {SUPPORT: [], CONTRADICT: [], NEI: []}
    for ex in raw:
        gt = get_ground_truth(ex.get("evidence", {}))
        by_label[gt].append({
            "id":            ex["id"],
            "claim":         ex["claim"],
            "ground_truth":  gt,
            "cited_doc_ids": ex.get("cited_doc_ids", []),
        })

    rng = random.Random(seed)
    per_class = n // 3
    sampled = []
    for i, label in enumerate([SUPPORT, CONTRADICT, NEI]):
        take = per_class + (1 if i < n % 3 else 0)
        pool = by_label[label][:]
        rng.shuffle(pool)
        sampled.extend(pool[:take])
    rng.shuffle(sampled)
    return sampled


# ── 4. FActScore ───────────────────────────────────────────────────────────────

def _factscore_verdict(decisions: list) -> tuple[str, float]:
    if not decisions:
        return NEI, 0.0
    ratio = sum(1 for d in decisions if d["is_supported"]) / len(decisions)
    if ratio >= FS_SUPPORT_THRESHOLD:
        return SUPPORT, ratio
    if ratio <= FS_CONTRADICT_THRESHOLD:
        return CONTRADICT, ratio
    return NEI, ratio


def run_factscore(claims: list, corpus: dict) -> list:
    db_path = build_scifact_db(corpus)

    fs = FactScorer(
        model_name="retrieval+ChatGPT",
        data_dir=FACTSCORE_DATA,
        cache_dir=str(SCIFACT_CACHE),
        openai_key=OPENAI_API_KEY,
    )
    db = DocDB(db_path=db_path)
    cache_path       = str(SCIFACT_CACHE / "retrieval-scifact.json")
    embed_cache_path = str(SCIFACT_CACHE / "retrieval-scifact.pkl")
    retrieval = Retrieval(db, cache_path, embed_cache_path, retrieval_type="bm25")
    fs.db["scifact"]        = db
    fs.retrieval["scifact"] = retrieval

    results = []
    for i, item in enumerate(claims):
        print(f"  [FActScore {i+1:3d}/{len(claims)}] {item['claim'][:65]}")

        topic = None
        for cid in item.get("cited_doc_ids", []):
            if cid in corpus:
                topic = corpus[cid]["title"]
                break

        if topic is None:
            results.append({**item, "fs_verdict": NEI,
                            "fs_decisions": [], "fs_topic": None, "fs_ratio": None})
            continue

        try:
            out = fs.get_score(
                [topic], [item["claim"]],
                gamma=10, knowledge_source="scifact"
            )
            decisions = out["decisions"][0] or []
            verdict, ratio = _factscore_verdict(decisions)
            results.append({
                **item,
                "fs_verdict":   verdict,
                "fs_decisions": decisions,
                "fs_topic":     topic,
                "fs_ratio":     round(ratio, 4),
            })
        except Exception as e:
            print(f"           → error: {e}")
            results.append({**item, "fs_verdict": NEI,
                            "fs_decisions": [], "fs_topic": topic, "fs_ratio": None})

    return results


# ── 5. uqlm ────────────────────────────────────────────────────────────────────

def _parse_label(text: str) -> str:
    t = text.strip().upper()
    head = t[:LABEL_SCAN_CHARS]

    if head.startswith("NOT_ENOUGH_INFO") or head.startswith("NOT ENOUGH"):
        return NEI
    if head.startswith("SUPPORTED"):
        return SUPPORT
    if head.startswith("CONTRADICTED"):
        return CONTRADICT

    if "NOT_ENOUGH_INFO" in head or "NOT ENOUGH INFO" in head:
        return NEI
    if "CONTRADICTED" in head:
        return CONTRADICT
    if "SUPPORTED" in head:
        return SUPPORT

    return NEI


async def run_uqlm(claims: list, corpus: dict) -> list:
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0.7)
    luq = LongTextUQ(llm=llm, scorers=["entailment"], response_refinement=False)

    prompts = []
    for c in claims:
        abstract_text = None
        for cid in c.get("cited_doc_ids", []):
            if cid in corpus:
                abstract_text = " ".join(corpus[cid]["abstract"])
                break

        if abstract_text:
            prompt = (
                f"Based ONLY on the following scientific abstract, evaluate the claim.\n\n"
                f"Abstract: {abstract_text}\n\n"
                f"Claim: {c['claim']}\n\n"
                f"Begin your response with exactly one label — "
                f"SUPPORTED, CONTRADICTED, or NOT_ENOUGH_INFO — then explain your reasoning."
            )
        else:
            prompt = (
                f"Evaluate the following scientific claim based on general evidence.\n\n"
                f"Claim: {c['claim']}\n\n"
                f"Begin with exactly one label: SUPPORTED, CONTRADICTED, or NOT_ENOUGH_INFO. "
                f"Then explain."
            )
        prompts.append(prompt)

    raw = await luq.generate_and_score(prompts=prompts, num_responses=3)
    df  = raw.to_df()

    results = []
    for i, (item, (_, row)) in enumerate(zip(claims, df.iterrows())):
        all_responses = []

        primary = row.get("response", "")
        if primary and isinstance(primary, str):
            all_responses.append(primary)

        sampled = row.get("sampled_responses", [])
        if isinstance(sampled, list):
            all_responses.extend(str(r) for r in sampled if r)

        parsed_labels = [_parse_label(r) for r in all_responses]
        if parsed_labels:
            count = Counter(parsed_labels)
            verdict, n_agree = count.most_common(1)[0]
            confidence = round(n_agree / len(parsed_labels), 3)
        else:
            verdict, confidence = NEI, 0.0
            count = Counter()

        try:
            cd = row["claims_data"] or []
        except (KeyError, TypeError):
            cd = []

        if cd:
            scores = [c.get("entailment", c.get("score", 0.5)) for c in cd]
            avg_entailment = round(sum(scores) / len(scores), 4)
        else:
            avg_entailment = 0.5

        print(
            f"  [uqlm     {i+1:3d}/{len(claims)}] vote={verdict}({confidence:.0%}) "
            f"entail={avg_entailment:.3f}  ({item['claim'][:45]})"
        )
        results.append({
            "id":                  item["id"],
            "uqlm_verdict":        verdict,
            "uqlm_confidence":     confidence,
            "uqlm_avg_entailment": avg_entailment,
            "uqlm_label_counts":   dict(count),
            "uqlm_claims_data":    cd,
        })

    return results


# ── 6. Compute metrics (no printing) ──────────────────────────────────────────

def _extract_class_metrics(report: dict, label: str) -> dict:
    row = report.get(label, {})
    return {
        "recall": round(row.get("recall", 0.0), 4),
        "f1":     round(row.get("f1-score", 0.0), 4),
    }


def compute_metrics(combined: list) -> dict:
    gt      = [c["ground_truth"] for c in combined]
    fs_pred = [c["fs_verdict"]   for c in combined]
    uq_pred = [c["uqlm_verdict"] for c in combined]
    labels  = [SUPPORT, CONTRADICT, NEI]

    fs_report = classification_report(gt, fs_pred, labels=labels,
                                      output_dict=True, zero_division=0)
    uq_report = classification_report(gt, uq_pred, labels=labels,
                                      output_dict=True, zero_division=0)

    metrics = {
        "n_claims":     len(combined),
        "distribution": {l: sum(1 for c in combined if c["ground_truth"] == l)
                         for l in labels},
        "factscore": {
            "accuracy":   round(accuracy_score(gt, fs_pred), 4),
            "macro_f1":   round(fs_report["macro avg"]["f1-score"], 4),
            "support":    _extract_class_metrics(fs_report, SUPPORT),
            "contradict": _extract_class_metrics(fs_report, CONTRADICT),
            "nei":        _extract_class_metrics(fs_report, NEI),
        },
        "uqlm": {
            "accuracy":   round(accuracy_score(gt, uq_pred), 4),
            "macro_f1":   round(uq_report["macro avg"]["f1-score"], 4),
            "support":    _extract_class_metrics(uq_report, SUPPORT),
            "contradict": _extract_class_metrics(uq_report, CONTRADICT),
            "nei":        _extract_class_metrics(uq_report, NEI),
        },
    }
    return metrics


# ── 7. Save / load ─────────────────────────────────────────────────────────────

def _serialize(o):
    if isinstance(o, dict): return {k: _serialize(v) for k, v in o.items()}
    if isinstance(o, list): return [_serialize(v) for v in o]
    if hasattr(o, "item"):  return o.item()
    return o


def save_results(combined: list, metrics: dict):
    (RESULTS_DIR / "level1_results.json").write_text(
        json.dumps(_serialize(combined), indent=2)
    )
    (RESULTS_DIR / "level1_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )


def load_saved_results() -> list:
    path = RESULTS_DIR / "level1_results.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No saved results at {path}. Run with rerun=True to generate them."
        )
    return json.loads(path.read_text())


# ── Entry point (callable from main script) ───────────────────────────────────

async def run_level1(rerun: bool = False) -> dict:
    """
    Run Level 1 (FActScore + uqlm) on 150 SciFact claims.

    rerun=False (default): load results/level1_results.json and recompute metrics.
    rerun=True:            re-run the full pipeline via the OpenAI API and overwrite
                           saved results. Requires OPENAI_API_KEY to be set.

    Returns the metrics dict. Also writes:
      results/level1_results.json
      results/level1_metrics.json
    """
    print("=== Level 1: FActScore + uqlm (150 claims) ===")

    if not rerun:
        print("Loading saved results (pass rerun=True to re-run the full pipeline)...")
        combined = load_saved_results()
        print(f"  {len(combined)} records loaded from results/level1_results.json")
        metrics = compute_metrics(combined)
        # Overwrite metrics file in case it's stale
        (RESULTS_DIR / "level1_metrics.json").write_text(json.dumps(metrics, indent=2))
        print("Metrics -> results/level1_metrics.json")
        return metrics

    # ── Full pipeline ──────────────────────────────────────────────────────────
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Export it before running with rerun=True."
        )

    corpus = load_corpus()
    print(f"Corpus loaded: {len(corpus)} documents")

    claims = load_claims(N_CLAIMS, RAND_SEED)
    dist = {l: sum(1 for c in claims if c["ground_truth"] == l)
            for l in [SUPPORT, CONTRADICT, NEI]}
    print(f"Claims loaded: {len(claims)} — {dist}")

    print("\n[1/2] Running FActScore...")
    fs_results = run_factscore(claims, corpus)

    print("\n[2/2] Running uqlm...")
    uqlm_results = await run_uqlm(claims, corpus)

    uqlm_by_id = {r["id"]: r for r in uqlm_results}
    combined = []
    for fs in fs_results:
        uq = uqlm_by_id[fs["id"]]
        combined.append({
            **fs, **uq,
            "agree": fs["fs_verdict"] == uq["uqlm_verdict"],
        })

    metrics = compute_metrics(combined)
    save_results(combined, metrics)

    print(f"\nLevel 1 complete. Results -> results/level1_results.json")
    print(f"Metrics -> results/level1_metrics.json")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Level 1: FActScore + uqlm on SciFact")
    parser.add_argument(
        "--rerun", action="store_true",
        help="Re-run the full pipeline via the API instead of loading saved results.",
    )
    args = parser.parse_args()
    asyncio.run(run_level1(rerun=args.rerun))