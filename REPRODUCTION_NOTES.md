# TreeQA: Implementation and Paper Comparison

## Project claim and scope

This repository is a resource-constrained prototype inspired by the TreeQA
paper. It implements the paper's broad workflow—logic-tree decomposition,
Wikidata/Wikipedia retrieval, evidence-based verification, subtree correction,
and answer generation—but it is **not a full reproduction** of the paper's
models, retrieval stack, datasets, or benchmark-scale evaluation. The local
experiment asks whether full TreeQA can improve over direct answering with the
same Groq-hosted model; the three-question pilot is descriptive only.

## What is implemented

### 1. Logic-tree construction

- `LogicTree.logic_tree_init` asks the configured LLM for a JSON tree with
  subquestions and concrete hypothesis answers.
- The prompt asks for pyramid-style organization and MECE-like children.
  MECE is a prompt instruction, not a mechanically checked property.
- Default limits are depth 3 and branching factor 4. Oversized generated
  trees are pruned to those limits.
- The final-answer call is instructed to return only a short answer string.

### 2. Entity linking and hybrid retrieval

- Entity mentions are extracted by an LLM. The current Wikidata configuration
  uses Wikidata search candidates and LLM-based candidate selection; it does
  not use Relik.
- Wikidata relation candidates are truncated to at most 40 per direction by
  default, then a Groq LLM selects relevant properties. Selected graph facts
  are provided as evidence.
- Wikipedia article sections are fetched and split into sentence-preserving
  chunks. The default ranker is local lexical word overlap. At most three
  chunks are returned, with a configured 220-word chunk limit (an approximate
  300-token target, not a tokenizer-enforced limit).
- Optional dense retrieval is implemented with CPU
  `sentence-transformers/all-MiniLM-L6-v2` embeddings and ChromaDB cosine
  ranking. It is opt-in (`TREEQA_RETRIEVE_MODEL_NAME=minilm` and
  `TREEQA_CHROMA_STORE=true`) and was smoke-tested locally; the pilot below
  used the default lexical ranker.
- Retrieval traces record query text, entity candidates and selected IDs,
  graph facts, and Wikipedia passages.

### 3. Verification and self-correction

- Full TreeQA checks generated nodes against retrieved text and graph evidence.
  The LLM classifies evidence as supported, contradicted, or unknown.
- Contradicted nodes can have their subtree rewritten. The rewritten node is
  retrieved and verified again before traversal continues.
- Defaults cap rewriting at one rewrite per node and ten tree updates per
  question. Unknown evidence permits at most one additional retrieval attempt.
- Verification, correction, and re-verification events are stored in the
  trace. These are LLM judgments over retrieved evidence, not a formal proof
  that the answer is true.

### 4. Groq model integration

- The local run uses the Groq OpenAI-compatible API with configured model
  `openai/gpt-oss-120b` and the local `GROQ_API_KEY`.
- The same configured model is used for DR, TreeQA without correction, and
  full TreeQA in the comparative harness.
- 429 responses are retried with backoff in the evaluation harness. Retries
  respect a `Retry-After` header when available; this handles transient rate
  limits but cannot bypass account quotas.

### 5. 2Wiki evaluation harness

`eval_2wiki.py` reads the repository's 1,000-row
`dataset/2wikiMultihopQA/dev_sampled.json`, saves a fixed seed-42 sample, and
supports resumable JSONL output and parsed Wikidata/Wikipedia response caches.
The three configurations are:

1. **DR** — direct short-answer generation, one model call, no tree/retrieval.
2. **TreeQA without correction** — builds a tree, retrieves evidence, and
   generates a final answer, but skips verification and rewrites.
3. **Full TreeQA** — runs tree construction, evidence retrieval, verification,
   correction, re-verification, and final answer generation.

For the time-limited run, the evaluation was restricted to the first three
questions of the fixed 20-question seed-42 sample. All three configurations
completed for all three questions: **9/9 successful runs**. This tiny sample is
not sufficient for a reliable performance conclusion.

The evaluator reports:

- **Hits@1**: normalized exact match for the single generated answer, using
  lowercase, punctuation/article removal, and whitespace normalization.
- **Token F1**: token overlap F1 after the same normalization.
- **Containment**: normalized reference substring match, a secondary diagnostic.
- Mean tokens/time/tree updates, Wilson intervals for binary metrics, bootstrap
  intervals for scalar metrics, and paired bootstrap differences for full
  TreeQA minus DR. The intervals are especially uncertain at N=3.
- Per-run question, gold answer, raw/short answer, metrics, model token count,
  wall time, updates, retries, trace, and error are checkpointed in JSONL.

### 6. Local demonstration app

Run `python app.py` and open `http://127.0.0.1:8501`.

- **Test Results** separates the saved 2Wiki comparative pilot from live QALD
  batch tests. The 2Wiki panel shows progress, per-configuration metrics,
  paired differences, question-level answers/traces, and JSON export. The QALD
  test controls can start fresh live runs; results are persisted separately.
- **Earlier Results** displays the existing 333-record historical QALD file
  and its summary. These records are historical outputs, not fresh calls or
  2Wiki results.
- **Live Inference** accepts a single question and displays a fresh answer,
  runtime/token metrics, and the module trace: initial tree, retrieval
  candidates/passages/facts, verification, and any corrections.
- Live calls check configuration first. Provider/retrieval errors are shown as
  failures; saved predictions are not substituted as fallbacks.

The UI was exercised on a fresh local server. Test Results displayed the
completed 2Wiki pilot and live QALD controls; Earlier Results loaded all 333
historical rows with 320 successful and 13 failed/missing records; Live
Inference rendered its question input and run control. The visible historical
summary was 66.6% containment, 0.0% exact match, 324.5 / 262.2 seconds
mean/median runtime, 17,795 / 14,071 mean/median tokens, and 0.29 mean
corrections. These are descriptive statistics for that saved historical file,
not the current live runs or the 2Wiki pilot. On Test Results, the 2Wiki
refresh loaded 9/9 saved runs, and selecting a question changed the displayed
trace; the QALD run/question selectors and trace sections were present. API
tests cover the 2Wiki results response and single-inference success path with
mocked inference. No additional paid/live model request was made during this
UI verification.

## Results from the completed three-question pilot

| Configuration | Hits@1 | F1 | Containment | Mean tokens/question | Mean seconds/question | Mean tree updates |
|---|---:|---:|---:|---:|---:|---:|
| DR | 2/3 (66.7%) | 0.667 | 2/3 (66.7%) | 344 | 1.4 | 0 |
| TreeQA without correction | 2/3 (66.7%) | 0.778 | 2/3 (66.7%) | 6,990 | 46.5 | 0 |
| Full TreeQA | 3/3 (100%) | 1.000 | 3/3 (100%) | 23,347 | 144.5 | 1.67 |

Paired full TreeQA minus DR difference: Hits@1 **+0.333** (bootstrap 95% CI
**[0.000, 1.000]**, paired N=3); F1 difference **+0.333** (95% CI
**[0.000, 1.000]**). The sample is too small and the intervals too wide to
claim a reliable improvement trend.

On one question, no-correction TreeQA returned “24 July 1229” (EM 0, F1
0.333), while full TreeQA identified conflicting evidence and returned “22
October 1229” (EM 1, F1 1.0). On the other two questions, DR was already
correct. This is an illustrative correction example, not aggregate evidence.

Full TreeQA used about **68 times** DR's mean tokens and took about **103 times**
DR's mean runtime in this pilot. The configured resource limits were not
evaluated against a comparable pre-change TreeQA baseline; therefore, no
before/after token savings are claimed.

## Paper setup vs. this implementation

| Dimension | TreeQA paper | This project |
|---|---|---|
| Goal | Benchmark experiments and ablations across reported LLM backbones. | Resource-constrained prototype; tests the same-model DR versus TreeQA comparison on a tiny pilot. |
| Backbone | Reported results include Qwen2.5-14B and GPT-3.5-turbo, among others. | Groq-hosted `openai/gpt-oss-120b`; not the paper's reported backbones. |
| Logic tree | Pyramid/MECE-guided decomposition and recursive refinement. | Similar prompt-guided structure and bounded recursion. MECE is not formally validated. |
| Entity linking | Relik-Entity-Linking-Large combined with LLM entity recognition. | LLM entity extraction/filtering with Wikidata search candidates. Relik is not installed or used. |
| Relation linking | Paper evaluates relation-retrieval methods including DeepSeek-V3 and NV-Embed-V2. | Groq selects properties from truncated Wikidata relation lists. Paper-style relation Recall@k is not measured. |
| Text retrieval | Dense Wikipedia retrieval using the paper's embedding setup. | Lexical ranking is the experiment default. Optional MiniLM + Chroma exists but was not used in this pilot. |
| Evidence/retrieval quality | Separate linking/retrieval experiments and benchmark analysis. | Candidate IDs and evidence are logged for inspection; entity-linking recall and retrieval recall are not calculated. |
| Dataset/protocol | Full benchmark dataset protocols and paper-specific evaluation. | Local 1,000-example sampled 2Wiki file; only the first 3 of a fixed seed-42 sample were evaluated in all configurations. |
| QA metric | Hits@1 reported in the paper's main QA comparison. | Normalized exact match is used as a single-answer Hits@1 proxy, plus token F1 and containment. It is not a full reproduction of the paper's scoring protocol. |
| Statistical strength | Results over the paper's evaluation sets and model settings. | N=3; paired bootstrap/Wilson intervals are shown but are extremely uncertain. No significance or generalization claim is warranted. |
| Cost/latency | Paper's reported model and pipeline costs. | Groq token usage and wall time measured locally; service rate limits affected elapsed time. |

The paper-reported 2Wiki results provided for reference are:

| Backbone | DR Hits@1 | TreeQA Hits@1 |
|---|---:|---:|
| Qwen2.5-14B | 43.2% | 51.5% |
| GPT-3.5-turbo | 54.6% | 59.0% |

These are the paper's numbers, **not this project's measurements**. The local
three-question outcome points in the same direction (3/3 full TreeQA versus
2/3 DR), but it is not enough to state that the paper's improvement trend has
been reproduced.

## What remains different or unverified

- Relik-Entity-Linking-Large is not used; the lightweight Wikidata/LLM flow is
  an explicit implementation difference.
- MiniLM dense retrieval is optional and smoke-tested, but lexical retrieval
  was used for the pilot. No QA-quality comparison between rankers exists.
- There is no paper-scale dataset run, multi-seed evaluation, or robust
  statistical test. N=3 is only a demonstrator.
- Entity-linking Recall@k, relation-linking Recall@k, retrieval recall/quality
  ablations, and the paper's error-category breakdown were not reproduced.
- No Precision@3 is reported: the system produces one answer, not three ranked
  candidate answers with relevance labels.
- The live inference UI depends on Groq quota and public Wikidata/Wikipedia
  availability. Retries cannot overcome exhausted account quotas.
- `requirements.txt` pins ChromaDB 0.6.3, whereas the current local environment
  was smoke-tested with ChromaDB 1.5.9 and sentence-transformers 6.2.0. Verify
  the optional embedding mode in the deployment environment.

## Reproduction commands

```powershell
# Local faculty demo
python app.py

# Display the saved 3-question comparative pilot
python eval_2wiki.py --summary result\2wiki_eval_seed42_n20.jsonl

# Resume the same bounded sample if a future run is needed
python eval_2wiki.py --run --n 20 --limit-questions 3 --seed 42

# Focused automated tests
python -m unittest tests.test_demo_data -v
```
