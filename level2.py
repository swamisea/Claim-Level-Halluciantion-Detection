"""
Level 2: FActScore with rationale gate + uqlm with label prompting (n=5)
on SciFact (150 balanced claims).

Improvements over Level 1:
  1. Rationale-gated FActScore — BM25 sentence relevance check before any API
     call; claims without sufficient evidence are declared NEI early.
     Also uses improved verdict logic: NEI-atom fraction threshold and a
     small-sample CONTRADICT adjustment for claims with <= 2 definitive atoms.
  2. uqlm with explicit label prompting — structured SUPPORTED / CONTRADICTED /
     NOT_ENOUGH_INFO prompt + 5 sampled responses for a more stable majority vote.

This script is designed to be called from a main runner script:

    from level2_scifact import run_level2

    # Default: load previously saved results (no API calls)
    metrics = asyncio.run(run_level2())

    # Force a fresh run (requires OPENAI_API_KEY)
    metrics = asyncio.run(run_level2(rerun=True))

CLI usage:
    python level2_scifact.py           # load saved results
    python level2_scifact.py --rerun   # re-run full pipeline

Returns a metrics dict and writes two files:
  results/level2_results.json   — per-sample predictions and raw outputs
  results/level2_metrics.json   — accuracy, macro_f1, and per-class recall/f1
                                   for FActScore and uqlm
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

import numpy as np
from rank_bm25 import BM25Okapi

sys.path.insert(0, "FActScore")

from sklearn.metrics import accuracy_score, classification_report
from langchain_openai import ChatOpenAI
from uqlm import LongTextUQ
from factscore.factscorer import FactScorer
from factscore.retrieval import DocDB, Retrieval

from dotenv import load_dotenv
load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
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

# FActScore support-ratio thresholds
FS_SUPPORT_THRESHOLD    = 0.6
FS_CONTRADICT_THRESHOLD = 0.4

# If > this fraction of atoms are "Neither" (insufficient context) -> NEI
FS_NEI_ATOM_FRAC = 0.60

# For claims with <= 2 definitive atoms: ratio <= this -> CONTRADICT
FS_SMALL_SAMPLE_CONTRADICT_THRESHOLD = 0.50

# Rationale gate: max sentence BM25 score below this -> NEI without API call
RELEVANCE_GATE_THRESHOLD = 0.5

# uqlm: sampled responses per claim
UQLM_NUM_RESPONSES = 5

LABEL_SCAN_CHARS = 80


# ── 1. Load corpus & claims ───────────────────────────────────────────────────

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


# ── 2. BM25 sentence scorer (rationale gate) ─────────────────────────────────

class CorpusRetriever:
    """BM25 sentence scorer for the rationale gate."""

    def __init__(self, corpus: dict):
        self.corpus = corpus

    def sentence_scores(self, claim: str, sentences: list) -> list:
        if not sentences:
            return []
        q_tokens  = claim.lower().split()
        tokenized = [s.lower().split() for s in sentences]

        if len(sentences) >= 3:
            scores = [float(s) for s in BM25Okapi(tokenized).get_scores(q_tokens)]
        else:
            q_set  = set(q_tokens)
            scores = [float(len(q_set & set(toks))) for toks in tokenized]

        return scores


# ── 3. Rationale-gated FActScore ─────────────────────────────────────────────

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


def _factscore_verdict(decisions: list) -> tuple:
    if not decisions:
        return NEI, 0.0

    nei_atoms = [d for d in decisions if d["is_supported"] is None]
    def_atoms = [d for d in decisions if d["is_supported"] is not None]
    nei_frac  = len(nei_atoms) / len(decisions)

    if nei_frac > FS_NEI_ATOM_FRAC or not def_atoms:
        return NEI, 0.0

    ratio = sum(1 for d in def_atoms if d["is_supported"]) / len(def_atoms)

    if len(def_atoms) <= 2:
        if ratio >= FS_SUPPORT_THRESHOLD:
            return SUPPORT, ratio
        if ratio <= FS_SMALL_SAMPLE_CONTRADICT_THRESHOLD:
            return CONTRADICT, ratio
        return NEI, ratio
    else:
        if ratio >= FS_SUPPORT_THRESHOLD:
            return SUPPORT, ratio
        if ratio <= FS_CONTRADICT_THRESHOLD:
            return CONTRADICT, ratio
        return NEI, ratio


def run_factscore(claims: list, corpus: dict, scorer: CorpusRetriever,
                  api_key: str) -> list:
    db_path = build_scifact_db(corpus)
    fs = FactScorer(
        model_name="retrieval+ChatGPT",
        data_dir=FACTSCORE_DATA,
        cache_dir=str(SCIFACT_CACHE),
        openai_key=api_key,
    )
    db = DocDB(db_path=db_path)
    cache_path       = str(SCIFACT_CACHE / "retrieval-scifact.json")
    embed_cache_path = str(SCIFACT_CACHE / "retrieval-scifact.pkl")
    retrieval = Retrieval(db, cache_path, embed_cache_path, retrieval_type="bm25")
    fs.db["scifact"]        = db
    fs.retrieval["scifact"] = retrieval

    results = []
    for i, item in enumerate(claims):
        claim = item["claim"]
        print(f"  [FActScore {i+1:3d}/{len(claims)}] {claim[:65]}")

        topic     = None
        sentences = []
        for cid in item.get("cited_doc_ids", []):
            if cid in corpus:
                topic     = corpus[cid]["title"]
                sentences = corpus[cid]["abstract"]
                break

        if topic is None:
            results.append({**item, "fs_verdict": NEI, "fs_decisions": [],
                            "fs_topic": None, "fs_ratio": None,
                            "fs_max_sent_score": 0.0, "fs_gated": True})
            continue

        # Rationale gate
        sent_scores    = scorer.sentence_scores(claim, sentences)
        max_sent_score = max(sent_scores) if sent_scores else 0.0

        if max_sent_score < RELEVANCE_GATE_THRESHOLD:
            results.append({**item, "fs_verdict": NEI, "fs_decisions": [],
                            "fs_topic": topic, "fs_ratio": None,
                            "fs_max_sent_score": round(max_sent_score, 3),
                            "fs_gated": True})
            continue

        try:
            out       = fs.get_score([topic], [claim], gamma=10,
                                     knowledge_source="scifact")
            decisions = out["decisions"][0] or []
            verdict, ratio = _factscore_verdict(decisions)
            results.append({
                **item,
                "fs_verdict":        verdict,
                "fs_decisions":      decisions,
                "fs_topic":          topic,
                "fs_ratio":          round(ratio, 4),
                "fs_max_sent_score": round(max_sent_score, 3),
                "fs_gated":          False,
            })
        except Exception as e:
            print(f"           -> error: {e}")
            results.append({**item, "fs_verdict": NEI, "fs_decisions": [],
                            "fs_topic": topic, "fs_ratio": None,
                            "fs_max_sent_score": round(max_sent_score, 3),
                            "fs_gated": False})

    return results


# ── 4. uqlm with explicit label prompting (n=5) ───────────────────────────────

_LABEL_TEMPLATE = (
    "Based ONLY on the following scientific abstract, evaluate the claim.\n\n"
    "Abstract: {abstract}\n\n"
    "Claim: {claim}\n\n"
    "Begin your response with exactly one label — SUPPORTED, CONTRADICTED, or "
    "NOT_ENOUGH_INFO — then explain your reasoning."
)

_LABEL_NO_ABSTRACT = (
    "Evaluate the following scientific claim based on general evidence.\n\n"
    "Claim: {claim}\n\n"
    "Begin with exactly one label: SUPPORTED, CONTRADICTED, or NOT_ENOUGH_INFO. "
    "Then explain."
)


def _parse_label(text: str) -> str:
    t    = text.strip().upper()
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


async def run_uqlm(claims: list, corpus: dict, api_key: str) -> list:
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0.7)
    luq = LongTextUQ(llm=llm, scorers=["entailment"], response_refinement=False)

    prompts = []
    for c in claims:
        abstract_text = None
        for cid in c.get("cited_doc_ids", []):
            if cid in corpus:
                abstract_text = " ".join(corpus[cid]["abstract"])
                break

        if abstract_text:
            prompts.append(_LABEL_TEMPLATE.format(abstract=abstract_text, claim=c["claim"]))
        else:
            prompts.append(_LABEL_NO_ABSTRACT.format(claim=c["claim"]))

    raw = await luq.generate_and_score(prompts=prompts, num_responses=UQLM_NUM_RESPONSES)
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
            count      = Counter(parsed_labels)
            verdict, n = count.most_common(1)[0]
            confidence = round(n / len(parsed_labels), 3)
        else:
            verdict, confidence = NEI, 0.0
            count = Counter()

        try:
            cd = row["claims_data"] or []
        except (KeyError, TypeError):
            cd = []

        avg_entailment = (
            round(sum(c.get("entailment", c.get("score", 0.5)) for c in cd) / len(cd), 4)
            if cd else 0.5
        )

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


# ── 5. Compute metrics (no printing) ─────────────────────────────────────────

def _extract_class_metrics(report: dict, label: str) -> dict:
    row = report.get(label, {})
    return {
        "recall": round(row.get("recall", 0.0), 4),
        "f1":     round(row.get("f1-score", 0.0), 4),
    }


def _system_metrics(gt: list, pred: list) -> dict:
    labels = [SUPPORT, CONTRADICT, NEI]
    report = classification_report(gt, pred, labels=labels,
                                   output_dict=True, zero_division=0)
    return {
        "accuracy":   round(accuracy_score(gt, pred), 4),
        "macro_f1":   round(report["macro avg"]["f1-score"], 4),
        "support":    _extract_class_metrics(report, SUPPORT),
        "contradict": _extract_class_metrics(report, CONTRADICT),
        "nei":        _extract_class_metrics(report, NEI),
    }


def compute_metrics(combined: list) -> dict:
    gt      = [c["ground_truth"] for c in combined]
    fs_pred = [c["fs_verdict"]   for c in combined]
    uq_pred = [c["uqlm_verdict"] for c in combined]
    labels  = [SUPPORT, CONTRADICT, NEI]

    return {
        "n_claims":     len(combined),
        "distribution": {l: sum(1 for c in combined if c["ground_truth"] == l)
                         for l in labels},
        "factscore": _system_metrics(gt, fs_pred),
        "uqlm":      _system_metrics(gt, uq_pred),
    }


# ── 6. Save / load ────────────────────────────────────────────────────────────

def _serialize(o):
    if isinstance(o, dict): return {k: _serialize(v) for k, v in o.items()}
    if isinstance(o, list): return [_serialize(v) for v in o]
    if hasattr(o, "item"):  return o.item()
    return o


def save_results(combined: list, metrics: dict):
    (RESULTS_DIR / "level2_results.json").write_text(
        json.dumps(_serialize(combined), indent=2)
    )
    (RESULTS_DIR / "level2_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )


def load_saved_results() -> list:
    path = RESULTS_DIR / "level2_results.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No saved results at {path}. Run with rerun=True to generate them."
        )
    return json.loads(path.read_text())


# ── Entry point (callable from main script) ───────────────────────────────────

async def run_level2(rerun: bool = False) -> dict:
    """
    Run Level 2 (gated FActScore + label-prompted uqlm) on 150 SciFact claims.

    rerun=False (default): load results/level2_results.json and recompute metrics.
    rerun=True:            re-run the full pipeline via the OpenAI API and overwrite
                           saved results. Requires OPENAI_API_KEY to be set.

    Returns the metrics dict. Also writes:
      results/level2_results.json
      results/level2_metrics.json
    """
    print("=== Level 2: Gated FActScore + Label-Prompted uqlm (150 claims) ===")

    if not rerun:
        print("Loading saved results (pass rerun=True to re-run the full pipeline)...")
        combined = load_saved_results()
        print(f"  {len(combined)} records loaded from results/level2_results.json")
        metrics = compute_metrics(combined)
        (RESULTS_DIR / "level2_metrics.json").write_text(json.dumps(metrics, indent=2))
        print("Metrics -> results/level2_metrics.json")
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
    dist   = {l: sum(1 for c in claims if c["ground_truth"] == l)
              for l in [SUPPORT, CONTRADICT, NEI]}
    print(f"Claims loaded: {len(claims)} -- {dist}")

    scorer = CorpusRetriever(corpus)

    print("\n[1/2] Running FActScore (rationale-gated)...")
    fs_results = run_factscore(claims, corpus, scorer, api_key)

    print("\n[2/2] Running uqlm (label prompting, 5 samples)...")
    uqlm_results = await run_uqlm(claims, corpus, api_key)

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

    print(f"\nLevel 2 complete. Results -> results/level2_results.json")
    print(f"Metrics -> results/level2_metrics.json")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Level 2: Gated FActScore + label-prompted uqlm on SciFact"
    )
    parser.add_argument(
        "--rerun", action="store_true",
        help="Re-run the full pipeline via the API instead of loading saved results.",
    )
    args = parser.parse_args()
    asyncio.run(run_level2(rerun=args.rerun))