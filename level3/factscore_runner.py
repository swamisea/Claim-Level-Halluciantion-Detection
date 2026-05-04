import os
from pathlib import Path
import sqlite3
from factscore.factscorer import FactScorer
from factscore.retrieval import DocDB, Retrieval
from bm25_retriever import CorpusRetriever
from dotenv import load_dotenv

load_dotenv()

class FActScoreRunner:
    def __init__(self, claims: list, corpus: dict, retriever: CorpusRetriever):
        self.claims = claims
        self.corpus = corpus
        self.retriever = retriever
        self.db_path = self.build_scifact_db(Path(os.environ.get("SCIFACT_DB")))
        self.db = DocDB(db_path=self.db_path)
        self.factscore = FactScorer(
            model_name="retrieval+ChatGPT",
            data_dir=os.path.expanduser(os.environ.get("FACTSCORE_DATA")),
            cache_dir=Path(os.environ.get("SCIFACT_CACHE")),
            openai_key=os.environ.get("OPENAI_API_KEY"),
        )
        self.fs_nei_atom_frac_threshold = float(os.environ.get("FS_NEI_ATOM_FRAC"))
        self.fs_support_threshold = float(os.environ.get("FS_SUPPORT_THRESHOLD"))
        self.fs_contradict_threshold = float(os.environ.get("FS_CONTRADICT_THRESHOLD"))
        self.fs_small_sample_contradict_threshold = float(os.environ.get("FS_SMALL_SAMPLE_CONTRADICT_THRESHOLD"))
        self.relevance_threshold = float(os.environ.get("RELEVANCE_GATE_THRESHOLD"))

        self.cache_path = str(Path(os.environ.get("SCIFACT_CACHE")) / "retrieval-scifact.json")
        self.embed_cache_path = str(Path(os.environ.get("SCIFACT_CACHE")) / "retrieval-scifact.pkl")

    def build_scifact_db(self, db_path: Path) -> Path:
        if db_path.exists():
            print("Scifact DB already exists")
            return db_path
        print(f"Building SciFact SQLite DB → {db_path}")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE documents (title PRIMARY KEY, text)")
        rows = [(doc["title"], doc["text"]) for doc in self.corpus.values()]
        conn.executemany("INSERT OR REPLACE INTO documents VALUES (?,?)", rows)
        conn.commit()
        conn.close()
        print(f"Created Scifact DB. {len(rows)} documents.")
        return db_path
    
    def factscore_verdict(self, decisions: list) -> tuple[str, float]:
        """
        3-way verdict using the modified FActScore prompt that outputs
        True / False / Neither (insufficient context).

        Key improvements over Level 2:
        - FS_NEI_ATOM_FRAC raised 0.40 → 0.60: a single uncertain atom out of 2
        no longer forces NEI when the remaining atom is definitively False.
        - Small-sample adjustment (≤ 2 definitive atoms): ratio ≤ 0.50 is treated
        as CONTRADICT instead of NEI, recovering the "1 false" pattern.
        """
        if not decisions:
            return "NEI", 0.0

        nei_atoms  = [d for d in decisions if d["is_supported"] is None]
        def_atoms  = [d for d in decisions if d["is_supported"] is not None]
        nei_frac   = len(nei_atoms) / len(decisions)

        if nei_frac > self.fs_nei_atom_frac_threshold:
            return "NEI", 0.0

        if not def_atoms:
            return "NEI", 0.0

        ratio = sum(1 for d in def_atoms if d["is_supported"]) / len(def_atoms)

        if len(def_atoms) <= 2:
            # With few atoms, be more aggressive on CONTRADICT (any false atom tips balance)
            if ratio >= self.fs_support_threshold:
                return "SUPPORT", ratio
            if ratio <= self.fs_small_sample_contradict_threshold:
                return "CONTRADICT", ratio
            return "NEI", ratio
        else:
            if ratio >= self.fs_support_threshold:
                return "SUPPORT", ratio
            if ratio <= self.fs_contradict_threshold:
                return "CONTRADICT", ratio
            return "NEI", ratio
    
    def run_factscore(self):
        print("--------- Running FActScore ---------")
        retrieval = Retrieval(self.db, self.cache_path, self.embed_cache_path, retrieval_type="bm25")
        self.factscore.db["scifact"] = self.db
        self.factscore.retrieval["scifact"] = retrieval
        results = []
        for i, item in enumerate(self.claims):
            claim = item["claim"]
            print(f"  [{i+1:2d}/{len(self.claims)}] {claim[:65]}")
            #Topic resolution: oracle first, open retrieval as fallback
            topic = None
            sentences = []
            source = "oracle"
            for cid in item.get("cited_doc_ids", []):
                if cid in self.corpus:
                    topic = self.corpus[cid]["title"]
                    sentences = self.corpus[cid]["abstract"]
                    break

            if topic is None:
                top_docs = self.retriever.retrieve(claim, k=1)
                best_doc = top_docs[0]
                topic = best_doc["title"]
                sentences = best_doc["abstract"]
                source = f"retrieved(score={best_doc['bm25_score']:.2f})"

            print(f"→ [{source}] \"{topic[:55]}\"")

            # ── Rationale gate: score sentences from the resolved abstract ───────
            sent_scores = self.retriever.sentence_scores(claim, sentences)
            max_sent_score = max(sent_scores) if sent_scores else 0.0

            if max_sent_score < self.relevance_threshold:
                print(f"→ gate FAIL (max_sent={max_sent_score:.2f}) → NEI")
                results.append({
                    **item,
                    "fs_verdict":        "NEI",
                    "fs_decisions":      [],
                    "fs_topic":          topic,
                    "fs_ratio":          None,
                    "fs_max_sent_score": round(max_sent_score, 3),
                    "fs_gated":          True,
                })
                continue

            print(f"→ gate OK (max_sent={max_sent_score:.2f})")
            print(f"Running Verfication for claim {i}")
            verification_results = self.run_verification(item, topic, claim, max_sent_score)
            results.append(verification_results)
        
        return results
    
    def run_verification(self, item, topic, claim, max_sent_score):
        try:
            out = self.factscore.get_score([topic], [claim], gamma=10, knowledge_source="scifact")
            decisions = out["decisions"][0] or []
            verdict, ratio = self.factscore_verdict(decisions)
            print(f"→ {len(decisions)} facts, ratio={ratio:.2f} → {verdict}")
            return {
                **item,
                "fs_verdict": verdict,
                "fs_decisions": decisions,
                "fs_topic": topic,
                "fs_ratio": round(ratio, 4),
                "fs_max_sent_score": round(max_sent_score, 3),
                "fs_gated": False,
            }
        except Exception as e:
            print(f"           → error: {e}")
            return {
                **item,
                "fs_verdict": "NEI",
                "fs_decisions": [],
                "fs_topic": topic,
                "fs_ratio": None,
                "fs_max_sent_score": round(max_sent_score, 3),
                "fs_gated": False,
            }
