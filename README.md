# Claim-Level Hallucination Detection

### Group 1:
- Swaminathan Chellappa
- Aaradhya Goyal
- Meeth Davda
- Aditya Sudhindra
- Karthik Venugopal

A three-level NLP pipeline for claim-level hallucination detection on the [SciFact](https://github.com/allenai/scifact) dataset. Each level progressively improves verdict accuracy by combining FActScore, uncertainty-quantified LLM scoring (uqlm), NLI, and ensemble methods to classify scientific claims as **SUPPORT**, **CONTRADICT**, or **NEI** (Not Enough Information).

## Table of Contents

- [Project Structure](#project-structure)
- [System & Device](#system--device)
- [Environment Setup](#environment-setup)
- [Running the Code](#running-the-code)
- [How Results Are Generated](#how-results-are-generated)
- [Results Summary](#results-summary)
- [Fine-tuning (TinyLlama + QLoRA)](#fine-tuning-tinyllama--qlora)

## Project Structure

```
.
├── main.py                        # Top-level runner: loads/reruns all levels and prints metrics
├── level1.py                      # Level 1: FActScore + uqlm baseline
├── level2.py                      # Level 2: BM25-gated FActScore + label-prompted uqlm (n=5)
├── level3/
│   ├── level3.py                  # Level 3: ensemble entry point
│   ├── factscore_runner.py        # Level 3 FActScore module
│   ├── uqlm_runner.py             # Level 3 uqlm module
│   ├── nli_runner.py              # GPT-based NLI module
│   ├── tinyllama_runner.py        # Fine-tuned TinyLlama NLI runner
│   ├── classifier_runner.py       # RandomForest score classifier
│   ├── bm25_retriever.py          # BM25 corpus retriever
│   ├── display_results.py         # Result formatting utilities
│   └── finetuned_tinyllama/       # Fine-tuned adapter weights
├── finetuning/
│   └── V3_MNLI_QLoRA_Colab.ipynb  # QLoRA fine-tuning notebook (Google Colab)
├── data/
│   └── scifact/                   # SciFact corpus, claims, and SQLite DB
├── results/                       # Saved JSON outputs and metrics per level
├── FActScore/                     # Local fork of FActScore
├── requirements.txt
├── requirements_no_deps.txt
└── .env.example                   # Template for environment variables
```

## System & Device

| Item | Details |
|---|---|
| OS | macOS (Darwin 25.x) |
| Python | 3.13 |
| Hardware (inference) | Apple Silicon Mac (CPU/MPS) |
| Hardware (fine-tuning) | Google Colab (NVIDIA GPU with CUDA, A100/T4 recommended) |
| LLM API | OpenAI `gpt-4o-mini` via LangChain |

## Environment Setup

### 1. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install main dependencies

```bash
pip install -r requirements.txt
pip install -r requirements_no_deps.txt --no-deps
```

### 3. Install FActScore separately (no-deps to avoid conflicts)

The `FActScore/` directory contains a local fork of the [FActScore](https://github.com/shmsw25/FActScore) library. Install it in editable mode without pulling in its pinned dependencies (which conflict with newer packages):

```bash
pip install -e FActScore/ --no-deps
```

### 4. Download the spaCy English model

```bash
python -m spacy download en_core_web_sm
```

### 5. Configure environment variables

Copy the example env file and fill in your credentials:

```bash
cp .env.example .env
```

Open `.env` and set at minimum:

```dotenv
OPENAI_API_KEY=sk-...          # Required for FActScore (ChatGPT) and uqlm
DATA_DIR=data/scifact/data
RESULTS_DIR=results
SCIFACT_DB=data/scifact/scifact_corpus.db
SCIFACT_CACHE=data/scifact/factscore_cache
FACTSCORE_DATA=~/.cache/factscore
N_CLAIMS=10
RAND_SEED=42
```

All other values in `.env.example` are pre-configured with defaults. Adjust the value for N_CLAIMS to increase or decrease the number of claims to be processed.

## Running the Code

### Display saved results (no API calls)

The `results/` directory contains pre-computed outputs. To print metrics without making any API calls:

```bash
python main.py
```

### Re-run all levels (makes OpenAI API calls)

```bash
python main.py --rerun
```

### Run specific levels only

```bash
# Run only Level 1 and Level 2
python main.py --levels 1 2

# Run only Level 3
python main.py --levels 3
```

### Run individual level scripts directly

```bash
# Level 1 - load saved results
python level1.py

# Level 1 - re-run full pipeline
python level1.py --rerun

# Level 2 - load saved results
python level2.py

# Level 2 - re-run full pipeline
python level2.py --rerun

# Level 3 - must be run from the level3/ directory
cd level3
python level3.py
```

## How Results Are Generated

The pipeline evaluates scientific claims from the SciFact training set against three labels: **SUPPORT**, **CONTRADICT**, and **NEI**. Each level adds capabilities over the previous one.

### Data

A balanced sample of claims is drawn from `data/scifact/data/claims_train.jsonl` with equal representation across SUPPORT, CONTRADICT, and NEI labels (controlled by `N_CLAIMS` and `RAND_SEED`). The SciFact corpus (`corpus.jsonl`) is loaded into an in-memory dictionary and also indexed in a SQLite database (`scifact_corpus.db`) for FActScore retrieval.

### Level 1 - FActScore + uqlm Baseline (`level1.py`)

Two independent systems score each claim:

1. **FActScore** (`retrieval+ChatGPT`): Decomposes the claim into atomic facts and uses BM25 retrieval over the SciFact corpus to check each atom against the cited abstract. The ratio of supported atoms is mapped to a verdict:
   - `ratio >= 0.6` → SUPPORT
   - `ratio <= 0.4` → CONTRADICT
   - otherwise → NEI

2. **uqlm** (`LongTextUQ` with entailment scorer): Sends the claim + abstract to `gpt-4o-mini` with `n=3` sampled responses. Each response is parsed for a leading label (`SUPPORTED` / `CONTRADICTED` / `NOT_ENOUGH_INFO`). The majority vote across samples becomes the final verdict.

Results are merged per claim and written to `results/level1_results.json` and `results/level1_metrics.json`.

### Level 2 - Gated FActScore + Label-Prompted uqlm (`level2.py`)

Improvements over Level 1:

1. **BM25 relevance gate**: Before calling the OpenAI API, a BM25 retriever scores each cited abstract against the claim. Claims with a maximum BM25 score below `RELEVANCE_GATE_THRESHOLD` are declared NEI immediately, saving API calls.

2. **Improved FActScore verdict logic**: Adds a NEI-atom fraction threshold (`FS_NEI_ATOM_FRAC`) - if most atoms are unscorable, the verdict defaults to NEI. A small-sample CONTRADICT adjustment tightens the threshold for claims with ≤ 2 definitive atoms.

3. **uqlm with explicit label prompting and n=5**: The prompt structure is more directive, and 5 sampled responses are used for a more stable majority vote.

Results are written to `results/level2_results.json` and `results/level2_metrics.json`.

### Level 3 - Ensemble (`level3/level3.py`)

Level 3 runs three sequential phases. Each phase uses a 3-signal ensemble of FActScore + uqlm + one NLI source - GPT and TinyLlama are alternatives, not used simultaneously.

**Phase 1 - FActScore + uqlm + GPT-4o-mini NLI**

- **FActScore** (same gated logic as Level 2, via `factscore_runner.py`)
- **uqlm** (label-prompted, n=5, via `uqlm_runner.py`)
- **GPT NLI** (`nli_runner.py`): Sends claim + top-K BM25 abstract sentences to `gpt-4o-mini` with a strict NLI system prompt. Maps ENTAILMENT → SUPPORT, CONTRADICTION → CONTRADICT, NEUTRAL → NEI.

The three verdicts are combined: if all agree the result is used directly; if two agree the majority wins; if all three disagree, a confidence-weighted tiebreak is applied (uqlm weight 0.45, NLI fixed 0.30, FActScore proportional to evidence quality). Phase 1 results are saved to `results/level3_results.json`.

**Phase 2 - TinyLlama NLI replaces GPT NLI**

Re-runs the same 3-signal ensemble but substitutes the GPT NLI call with a fine-tuned TinyLlama model (`tinyllama_runner.py`). The adapter is loaded from `level3/finetuned_tinyllama/` on top of `TinyLlama/TinyLlama-1.1B-Chat-v1.0` and generates a label token for each claim. Results are saved to `results/level3_tinyllama_results.json`. Phase 2 is skipped if `TINYLLAMA_MODEL` is not set in `.env`.

**Phase 3 - RandomForest Score Classifier**

A `RandomForest` classifier (`classifier_runner.py`) is trained on numeric features (FActScore ratio, uqlm entailment score, confidence) using 5-fold cross-validation. It always runs once with FActScore + uqlm features only (no NLI). If Phase 2 ran, it also runs a second time with TinyLlama NLI features added. 

A **RandomForest score classifier** (`classifier_runner.py`) is also available as an optional signal. It trains on numeric features (FActScore ratio, uqlm entailment score, confidence) using 5-fold cross-validation (`CV_N_SPLITS=5`) with `RF_N_ESTIMATORS=200` trees.

## Fine-tuning (TinyLlama + QLoRA)

The notebook `finetuning/V3_MNLI_QLoRA_Colab.ipynb` trains a QLoRA adapter on the [MultiNLI](https://huggingface.co/datasets/nyu-mll/multi_nli) dataset for claim verification. It is designed to run on Google Colab (requires a GPU).

Key settings:
- **Base model**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Quantization**: 4-bit NF4 with double quantization (`bitsandbytes`)
- **LoRA rank**: `r=16`, `alpha=32`, targeting all attention and MLP projection layers
- **Dataset**: 5,000 train / 1,000 eval examples from MultiNLI
- **Label mapping**: `entailment` → SUPPORT, `contradiction` → CONTRADICT, `neutral` → NOT_ENOUGH_INFO

The trained adapter is saved and placed at `level3/finetuned_tinyllama/` for use by `tinyllama_runner.py` at inference time.
