import copy
import os
import sys
import json
import random
import asyncio
from collections import Counter
from pathlib import Path
sys.path.insert(0, "FActScore")
from bm25_retriever import CorpusRetriever
from factscore_runner import FActScoreRunner
from uqlm_runner import UQLMRunner
from nli_runner import NLIRunner
from display_results import DisplayResults
from tinyllama_runner import TinyLlamaNLIRunner
from classifier_runner import Classifier
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent

Path(ROOT / os.environ.get("RESULTS_DIR")).mkdir(exist_ok=True)
Path(ROOT / os.environ.get("SCIFACT_CACHE")).mkdir(parents=True, exist_ok=True)

class Level3:
    def __init__(self):
        print("Setting up Level 3 ...")
        self.corpus = self.load_corpus()
        self.claims = self.load_claims(int(os.environ.get("N_CLAIMS")), int(os.environ.get("RAND_SEED")))
        self.retriever = CorpusRetriever(self.corpus)

    def load_corpus(self) -> dict:
        print("1. Loading SciFact corpus")
        corpus = {}
        for line in (ROOT / os.environ.get("DATA_DIR") / "corpus.jsonl").open():
            doc = json.loads(line)
            sentences = doc["abstract"]
            corpus[doc["doc_id"]] = {
                "title": doc["title"],
                "abstract": sentences,
                "text": os.environ.get("SPECIAL_SEPARATOR").join(sentences),
            }
        print(f"  {len(corpus)} documents loaded.")
        return corpus    
    
    def get_ground_truth(self, evidence: dict) -> str:
        if not evidence:
            return "NEI"
        labels = set()
        for v in evidence.values():
            for item in (v if isinstance(v, list) else [v]):
                if isinstance(item, dict):
                    labels.add(item.get("label", ""))
        if "SUPPORT" in labels: return "SUPPORT"
        if "CONTRADICT" in labels: return "CONTRADICT"
        return "NEI"
    
    def load_claims(self, n: int = 50, seed: int = 42) -> list:
        print("2. Loading SciFact claims")
        training_data = [json.loads(l) for l in (ROOT / os.environ.get("DATA_DIR") / "claims_train.jsonl").open()]

        by_label = {"SUPPORT": [], "CONTRADICT": [], "NEI": []}
        for example in training_data:
            gt = self.get_ground_truth(example.get("evidence", {}))
            by_label[gt].append({
                "id": example["id"],
                "claim": example["claim"],
                "ground_truth": gt,
                "cited_doc_ids": example.get("cited_doc_ids", []),
            })
    
        rng = random.Random(seed)
        per_class = n // 3
        sampled = []
        for i, label in enumerate(["SUPPORT", "CONTRADICT", "NEI"]):
            take = per_class + (1 if i < n % 3 else 0)
            pool = by_label[label][:]
            rng.shuffle(pool)
            sampled.extend(pool[:take])
        rng.shuffle(sampled)

        dist = {l: sum(1 for c in sampled if c["ground_truth"] == l)
                for l in ["SUPPORT", "CONTRADICT", "NEI"]}
        print(f"  {len(sampled)} claims — {dist}")
        return sampled

    def simple_ensemble_verdicts(self, fs_results: list, uqlm_results: list, nli_results: list | None = None) -> list:
        """
        Three-signal aggregator: FActScore + uqlm + NLI.

        Decision rules:
        1. All three agree           → use that verdict
        2. Two of three agree (majority) → use the majority verdict
        3. All three disagree (degenerate) → confidence-weighted:
            uqlm gets weight proportional to uqlm_confidence;
            FActScore gets weight from fs_max_sent_score (evidence quality);
            NLI gets fixed weight 0.3 (single call, no uncertainty estimate).

        When nli_results is None (two-signal mode), falls back to original logic:
        uqlm wins on confidence ≥ 0.6, else FActScore.
        """

        print("Combining results using a simple Ensemble technique")
        uqlm_by_id = {r["id"]: r for r in uqlm_results}
        nli_by_id  = {r["id"]: r for r in nli_results} if nli_results else {}

        results = []
        for fs in fs_results:
            uq  = uqlm_by_id[fs["id"]]
            nli = nli_by_id.get(fs["id"])

            fs_v  = fs["fs_verdict"]
            uq_v  = uq["uqlm_verdict"]
            nli_v = nli["nli_verdict"] if nli else None

            if nli_v is None:
                # Two-signal fallback (original logic)
                if fs_v == uq_v:
                    verdict, method = fs_v, "agree_2"
                elif uq["uqlm_confidence"] >= 0.6:
                    verdict, method = uq_v, "uqlm_wins"
                else:
                    verdict, method = fs_v, "fs_wins"
            else:
                votes = [fs_v, uq_v, nli_v]
                counts = Counter(votes)
                top_label, top_count = counts.most_common(1)[0]

                if top_count == 3:
                    verdict, method = top_label, "agree_3"
                elif top_count == 2:
                    verdict, method = top_label, "majority_2of3"
                else:
                    # All three disagree: confidence-weighted
                    fs_conf  = min(1.0, (fs.get("fs_max_sent_score") or 0) / 10.0) * 0.25
                    uq_conf  = uq["uqlm_confidence"] * 0.45
                    nli_conf = 0.30  # fixed weight for single NLI call
                    weights  = {fs_v: fs_conf, uq_v: uq_conf, nli_v: nli_conf}
                    verdict  = max(weights, key=weights.get)
                    method   = "conf_weighted"

            results.append({
                **fs,
                **uq,
                **({"nli_verdict": nli_v, "nli_raw": nli.get("nli_raw", "")} if nli else {}),
                "ensemble_verdict": verdict,
                "ensemble_method":  method,
            })
        return results
    
    async def run_pipeline(self) -> None:
        # ── Phase 1: Base pipeline ─────────────────────────────────────────────
        _header("PHASE 1 — Base Pipeline  (FActScore + UQLM + GPT-4o-mini NLI)")
        factscore_results = FActScoreRunner(
            claims=self.claims, corpus=self.corpus, retriever=self.retriever
        ).run_factscore()
        uqlm_results, gpt_nli_results = await asyncio.gather(
            UQLMRunner(claims=self.claims, corpus=self.corpus, retriever=self.retriever).run_uqlm(),
            NLIRunner(claims=self.claims, corpus=self.corpus, retriever=self.retriever).run_nli(),
        )
        base_results = self.simple_ensemble_verdicts(factscore_results, uqlm_results, gpt_nli_results)

        display = DisplayResults()
        display.evaluate_and_print(base_results)
        display.save_results(base_results)

        # ── Phase 2: TinyLlama NLI ─────────────────────────────────────────────
        _header("PHASE 2 — TinyLlama NLI Replacement")
        tinyllama_nli_by_id = None
        tinyllama_results   = None
        if os.environ.get("TINYLLAMA_MODEL"):
            runner           = TinyLlamaNLIRunner(corpus=self.corpus)
            tinyllama_nli    = runner.run_nli(base_results)
            tinyllama_nli_by_id = {r["id"]: r for r in tinyllama_nli}
            tinyllama_results = self.simple_ensemble_verdicts(
                factscore_results, uqlm_results, tinyllama_nli
            )
            display.evaluate_and_print(tinyllama_results)
            _save_json(
                tinyllama_results,
                ROOT / os.environ.get("RESULTS_DIR") / "level3_tinyllama_results.json",
            )
        else:
            print("  TINYLLAMA_MODEL not set in .env — skipping Phase 2")

        # ── Phase 3: Score Classifier ──────────────────────────────────────────
        _header("PHASE 3 — Score Classifier")
        clf = Classifier()

        clf_records_no_nli = copy.deepcopy(base_results)
        clf.run(clf_records_no_nli)
        clf.print_report("Classifier — FActScore + UQLM features only", clf_records_no_nli)
        clf_preds_no_nli = [r["classifier_verdict"] for r in clf_records_no_nli]

        clf_preds_with_nli = None
        if tinyllama_nli_by_id:
            clf_records_with_nli = copy.deepcopy(base_results)
            clf.run(clf_records_with_nli, nli_by_id=tinyllama_nli_by_id)
            clf.print_report("Classifier — FActScore + UQLM + TinyLlama NLI features", clf_records_with_nli)
            clf_preds_with_nli = [r["classifier_verdict"] for r in clf_records_with_nli]
            clf.save_results(clf_records_with_nli)
        else:
            clf.save_results(clf_records_no_nli)

        # ── Final comparison ───────────────────────────────────────────────────
        self._print_final_summary(base_results, tinyllama_results, clf_preds_no_nli, clf_preds_with_nli)

    def _print_final_summary(self, base_results, tinyllama_results, clf_preds_no_nli, clf_preds_with_nli):
        from sklearn.metrics import f1_score, accuracy_score as acc_fn

        gt   = [r["ground_truth"] for r in base_results]
        labs = ["SUPPORT", "CONTRADICT", "NEI"]

        def row(name, pred):
            if pred is None or any(v is None for v in pred):
                return
            acc = acc_fn(gt, pred)
            mf1 = f1_score(gt, pred, labels=labs, average="macro", zero_division=0)
            print(f"  {name:<44} {acc:>8.1%}  {mf1:>8.3f}")

        _header("FINAL SUMMARY — All Systems")
        print(f"  {'System':<44} {'Accuracy':>8}  {'Macro F1':>8}")
        print("  " + "-" * 64)
        row("FActScore (base)",              [r["fs_verdict"]       for r in base_results])
        row("UQLM (base)",                   [r["uqlm_verdict"]     for r in base_results])
        row("GPT-4o-mini NLI (base)",        [r.get("nli_verdict")  for r in base_results])
        row("Ensemble — GPT NLI (Phase 1)",  [r["ensemble_verdict"] for r in base_results])
        if tinyllama_results:
            print("  " + "-" * 64)
            row("TinyLlama NLI (Phase 2)",        [r.get("nli_verdict")  for r in tinyllama_results])
            row("Ensemble — TinyLlama (Phase 2)", [r["ensemble_verdict"] for r in tinyllama_results])
        if clf_preds_no_nli or clf_preds_with_nli:
            print("  " + "-" * 64)
        if clf_preds_no_nli:
            row("Classifier — no NLI (Phase 3)",     clf_preds_no_nli)
        if clf_preds_with_nli:
            row("Classifier + TinyLlama (Phase 3)",  clf_preds_with_nli)
        print("  " + "=" * 64)


# ── Module-level helpers ───────────────────────────────────────────────────────

def _header(title: str):
    print(f"\n{'='*70}")
    print(title)
    print("=" * 70)


def _save_json(data: list, path: Path):
    def serial(o):
        if isinstance(o, dict): return {k: serial(v) for k, v in o.items()}
        if isinstance(o, list): return [serial(v) for v in o]
        if hasattr(o, "item"):  return o.item()
        return o
    path.write_text(json.dumps(serial(data), indent=2))
    print(f"Saved → {path}")


if __name__ == "__main__":
    asyncio.run(Level3().run_pipeline())


    
