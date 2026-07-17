# Gen-AI Email Suggested-Response System

An end-to-end system that (1) drafts a suggested reply to an incoming email
using an LLM grounded in a dataset of past emails, and (2) the main focus
of this project **measures how good each suggestion actually is**, with
per-response and overall scores.

```
email_ai_system/
├── data/
│   ├── generate_dataset.py   # builds the dataset
│   └── dataset.jsonl         # 29 (email, reply, key_points) examples
├── src/
│   └── generator.py          # retrieval + prompting -> LLM reply
├── eval/
│   └── evaluate.py           # THE ACCURACY SYSTEM (core deliverable)
├── run_pipeline.py           # end-to-end demo over the whole dataset
├── requirements.txt
├── .env.example               # template for your GROQ_API_KEY
├── .gitignore
└── report.json                # produced by run_pipeline.py
```

## 1. The Dataset - Where it came from and Why it's Representative

`data/generate_dataset.py` produces `dataset.jsonl`: **29 hand-authored,
synthetic (incoming_email, reply) pairs** across 6 realistic inbox
categories : `scheduling`, `customer_support`, `sales_inquiry`,
`internal_hr`, `complaint_escalation`, `informational`.

**Why hand-authored/synthetic rather than a scraped public corpus (e.g.
Enron):**
- Real inboxes contain PII and confidential business info not something
  you can put in a public repo responsibly.
- Public email corpora (Enron, etc.) are messy, thread-heavy, and don't
  come with a *correct-reply checklist* and that checklist is the single
  most important thing this project needs (see §3). Hand-authoring lets us
  guarantee it exists and is accurate for every example.
- Hand-authoring lets us deliberately balance categories and behaviors
  (apology + remedy, schedule negotiation, escalation, information lookup)
  that a real business inbox sees, instead of whatever a scrape happens to
  contain.

**Why it's still representative:** each example is modeled directly on a
realistic real-world scenario (order delays, double billing, PTO requests,
pricing inquiries, angry customers, etc.) the kind of things any
support/sales/ops inbox handles daily. It is intentionally small (29
examples) because the point of this exercise is the *evaluation system*,
not scale; the dataset and pipeline are built to scale add more rows to
`generate_dataset.py` (or point `load_dataset()` at a different `.jsonl`
with the same schema) and everything downstream keeps working.

**Schema per example:**
```json
{
  "id": "support_001",
  "category": "customer_support",
  "subject": "Order hasn't arrived",
  "incoming_email": "...",
  "reply": "... the reply that was actually sent ...",
  "key_points": ["apologizes for the delay", "references order #4521", "..."]
}
```
`key_points` is the critical annotation: the minimal checklist a *correct*
reply must satisfy. It matters more than the literal reply text see §3.

## 2. The Generator (Gen-AI, not classical ML)

`src/generator.py`:
1. **Retrieves** the *k* most similar past emails from the dataset using
   TF-IDF cosine similarity over subject+body (`Retriever`).
2. **Builds a few-shot prompt** from those retrieved (email, reply) pairs
   plus the new incoming email (`build_prompt`).
3. **Calls Groq** (via the `groq` SDK, OpenAI-compatible chat completions)
   to generate the suggested reply, grounded in that retrieved context.

   **Model used: `llama-3.3-70b-versatile`** (default) strong quality,
   fast inference, and a generous free tier, which is why the project
   moved to Groq after hitting Google AI Studio's free-tier quota limits
   during testing. For a lighter/faster option with more headroom, swap in
   `llama-3.1-8b-instant` by passing `model="llama-3.1-8b-instant"` to
   `generate_reply()` / `run_pipeline.py`'s `generate_leave_one_out()`, or
   by editing the default in `src/generator.py`.

### Why retrieval + few-shot prompting instead of fine-tuning
- The dataset is intentionally small fine-tuning on 29 examples would
  overfit and add real cost/complexity for no benefit. A frontier LLM
  already knows how to write a good email; what it needs is *grounding* in
  this specific inbox's tone, facts, and past resolutions.
- Retrieval keeps the system **inspectable**: every generated reply can be
  traced back to the exact past examples that informed it which directly
  feeds the evaluation system (you can check whether the model actually
  used what it was given).
- **Trade-off:** at real scale (thousands of emails, a mature product) a
  hybrid approach makes more sense retrieval/prompting to start, with
  periodic light fine-tuning or preference-tuning on accepted vs.
  human-edited suggestions once there's enough signal to do it safely.

### Running without an API key
If `GROQ_API_KEY` isn't set, `generator.py` automatically falls back
to a deterministic, no-network "offline" generator (adapts the single
closest retrieved reply). This exists purely so the **whole pipeline is
runnable and gradeable without a key or network access** it is *not*
meant to represent generation quality; the LLM path is the intended one.

## 3. The Accuracy/Evaluation System - the Core of this Project

`eval/evaluate.py`. Full reasoning is in the module docstring; summary
below.

### What does "Accurate" mean here?
Exact string match against the historical reply is the wrong bar two
replies can use totally different words and both be great. Instead,
accuracy is decomposed into three checkable dimensions:

| # | Metric | Question it answers | How it's computed |
|---|--------|---------------------|--------------------|
| 1 | **Key-point coverage** (45%) | Did it actually address what a correct reply needs to? | Regex/pattern matching of each example's hand-authored `key_points` checklist against the generated reply |
| 2 | **Semantic similarity** (25%) | Is it on-topic and in a similar spirit to how we've handled this before? | TF-IDF cosine similarity vs. the historical reply |
| 3 | **LLM-as-judge** (30%, optional) | Would a human reviewer call this a good, safe, sendable reply? | Groq (`llama-3.3-70b-versatile`) scores 1-5 on correctness/tone/completeness with a justification |

These combine into a **composite score, 0–100, per response**. If no API
key is available, weight 3 is redistributed proportionally across 1 and 2
so the score stays on a consistent 0–100 scale.

**Why this combination and not just one number:** coverage catches "did it
do the job," similarity catches "does it sound like how we actually talk
to people," and the LLM judge catches things the other two structurally
can't hallucinated facts, wrong tone, a technically-keyword-matching but
nonsensical reply. Any one of these alone is gameable; together they're
much harder to fool (e.g. a reply that stuffs in every key-point keyword
but reads incoherently would still lose on the LLM-judge dimension).

### How we validate the metric reflects real quality (not just a number)
`validate_against_human_ratings()` in `evaluate.py` is a small, honest
sanity check: for 3 incoming emails we hand-write a **GOOD / MEDIOCRE /
BAD** reply each, with our own 1–5 human quality rating, then check whether
the composite score ranks them the same way the human rating does
(pairwise ranking agreement).

Run it: `python eval/evaluate.py`

Result on this dataset: **89% pairwise ranking agreement (8/9 pairs)** -
the one disagreement is disclosed in the output, not hidden: a "bad" reply
scored higher than a "mediocre" one because it happened to hit a keyword
pattern. This is an honest illustration of the known trade-off of
keyword-based coverage matching (see limitations below) at production
scale you'd validate against a larger sample of real human reviewer labels
and iterate on the pattern lists / consider swapping in embedding
similarity or a dedicated NLI-based checker for coverage.

### Reporting
`evaluate_dataset()` returns:
- **Per-response**: composite score, each sub-metric, which key points were
  covered/missed, and a short human-readable explanation string.
- **Overall system score**: mean composite score across all responses.
- **Category breakdown**: mean score per email category (some intents,
  e.g. `complaint_escalation`, are inherently harder than others, e.g.
  `informational` - reporting this separately avoids a single number
  hiding weak spots).

### A real end-to-end run (Groq, `llama-3.3-70b-versatile`)
```
Generation mode: LLM (Groq API)
...
CATEGORY BREAKDOWN
  scheduling              67.1/100
  customer_support        53.7/100
  sales_inquiry           67.0/100
  internal_hr             53.4/100
  complaint_escalation    58.3/100
  informational           49.6/100

OVERALL SYSTEM SCORE: 58.0/100  (n=29)
```
Two things worth calling out from this run, because they show the
evaluation system working as intended rather than just producing a number:
- Scores spread meaningfully (17–84) and track what a human reading the
  LLM-judge justifications would agree with — not clustered or random.
- A few examples (e.g. `support_004`) show the LLM judge scoring 5/5 while
  keyword coverage scored 0% — a real, disclosed limitation of regex-based
  coverage (the reply likely paraphrased the required points in wording
  the patterns didn't anticipate). This is exactly the kind of
  judge-vs-coverage disagreement the validation section above predicts,
  and the next concrete improvement is expanding `KEY_POINT_PATTERNS` for
  those cases.

## 4. Getting a Groq API key

1. Go to **https://console.groq.com/keys**.
2. Sign in, click **"Create API Key"**, and copy it (starts with `gsk_...`).
3. Check your current free-tier limits at
   **https://console.groq.com/settings/limits** — they're generous but do
   change over time, and the leave-one-out pipeline makes ~2 calls per
   dataset example (generation + judge), so ~58 calls for the full 29-row
   dataset.
4. Provide the key to the project using **either** of these (both work —
   pick whichever fits your workflow):

   **Option A — environment variable (no extra setup):**
   ```bash
   export GROQ_API_KEY="gsk_..."      # macOS/Linux
   set GROQ_API_KEY=gsk_...           # Windows cmd
   $env:GROQ_API_KEY="gsk_..."        # Windows PowerShell
   ```
   You'll need to re-set this every new terminal session.

   **Option B — `.env` file (recommended, persists across sessions):**
   ```bash
   cp .env.example .env
   # then edit .env and paste your real key:
   #   GROQ_API_KEY=gsk_...
   ```
   `python-dotenv` (already in `requirements.txt`) loads this automatically
   — no need to `export` anything. `.env` is already listed in
   `.gitignore` so it will never be committed.

   Never commit your real key or hardcode it in the code either way.

## 5. Running it end-to-end

```bash
cd email_ai_system
pip install -r requirements.txt

# (re)build the dataset — already checked in, but reproducible:
python data/generate_dataset.py

# set your Groq key (see §4), then run the full pipeline:
export GROQ_API_KEY="gsk_..."            # macOS/Linux
# set GROQ_API_KEY=gsk_...               # Windows (cmd)
# $env:GROQ_API_KEY="gsk_..."            # Windows (PowerShell)

python run_pipeline.py                   # full Gen-AI mode (Groq)

# or, without setting a key at all:
python run_pipeline.py                   # offline mode (no API calls)

# run just the metric-validation sanity check
python eval/evaluate.py
```

`run_pipeline.py` writes a full `report.json` with every score.

**Note on the offline-mode numbers:** the offline fallback generator is
intentionally simplistic (it lightly adapts the single closest retrieved
reply). Run with `GROQ_API_KEY` set to see real Gen-AI-quality suggestions
and scores — the low offline scores (~30/100) are the evaluation system
correctly doing its job, not a bug. With Groq active, the same dataset
scores ~58/100 overall (see §3's example run).

## 6. How AI tools were used
This project (dataset authoring, generator, evaluation system, and this
README) was built with Claude (Anthropic) as a pair-programmer: drafting
the synthetic dataset examples, writing/iterating on the retrieval +
prompting code, designing and implementing the multi-metric evaluation
system, and writing the validation harness. The generation and LLM-judge
components at runtime call the Groq API (`llama-3.3-70b-versatile`). All
code was run and verified in-session (dataset generation, offline pipeline
run, live Groq run, and the metric validation check all execute
successfully — see `report.json` for a real run's output).

## 7. Step-by-step guide to building this project

1. **Decide what "accurate" means before writing any code.** This is the
   hardest and most important part. We chose: coverage of required
   content > topical/style similarity > holistic LLM judgment.
2. **Design the dataset schema around the metric, not the other way
   around.** Adding `key_points` per example was the key decision — without
   it, evaluation degrades to fuzzy text similarity, which is weak.
3. **Hand-author a small, balanced dataset** across realistic categories
   (`data/generate_dataset.py`), each with a subject, incoming email, sent
   reply, and key-points checklist.
4. **Build retrieval** over the dataset (TF-IDF here; swap for embeddings
   for scale) so generation can be grounded in relevant past examples.
5. **Build the prompt + LLM call** that turns (new email + retrieved
   examples) into a suggested reply (`src/generator.py`), with an offline
   fallback so the system still runs without an API key.
6. **Build each evaluation sub-metric independently and testably**:
   coverage (`key_point_coverage`), similarity (`semantic_similarity`),
   LLM-judge (`llm_judge`) — each is a small, unit-testable function.
7. **Combine into a composite score** with explicit, justified weights, and
   make sure the system degrades gracefully (e.g. no API key -> redistribute
   weights) rather than crashing.
8. **Validate the metric** against a small hand-labeled good/mediocre/bad
   set and report the agreement honestly, including its limitations
   (`validate_against_human_ratings`).
9. **Wire it all into one runnable pipeline** (`run_pipeline.py`) that
   produces per-response scores, a category breakdown, and an overall
   score, plus a machine-readable `report.json`.
10. **Write the README** explaining the *why*, not just the *what*, for
    every major decision — the dataset source, the generation approach, and
    above all the accuracy metric design and its validation.

## 8. Known limitations / next steps
- Key-point coverage uses regex patterns, not true NLU — a correct reply
  phrased very differently from the anticipated patterns could be
  under-scored (observed directly in the real run above). Mitigated by the
  LLM-judge metric when available; a production version would likely use
  an NLI/entailment model per key point instead of regex.
- TF-IDF similarity is a weak proxy for semantic similarity; swapping in
  sentence embeddings would improve metric 1 with a network/compute
  trade-off.
- The metric-validation set (9 hand-labeled examples) is small — a real
  deployment should validate against dozens–hundreds of real reviewer
  labels before trusting the composite score for decisions like
  auto-sending replies above a threshold.
- The LLM judge itself hasn't been separately validated against human
  raters here beyond the same small sanity set — worth doing at scale
  since judge models have their own blind spots/biases.
- Free-tier LLM API quotas (Groq included) can change or be exhausted
  mid-run; `run_pipeline.py` reports how many calls fell back to offline
  generation so this is always visible rather than silently swallowed.

