# Evaluation Work Handoff

## Changes made

| File | Change |
|---|---|
| `treeQA/tree_class/logicTree.py` | Re-verifies a rewritten node with fresh retrieval before descending; records correction/re-verification outcomes; caps rewrites; limits generated tree depth/branching; requests concise hypotheses and final answers. |
| `treeQA_Config.py` | Centralized configurable limits for tree depth, branching, rewrites, total updates, retrieval retries, Wikipedia chunk size/top-k, and relation-list truncation. |
| `treeQA/wikipediaUtills.py` | Uses the configured ~300-token-equivalent chunk size (220 words), returns up to three chunks, and supports optional MiniLM embeddings stored and cosine-ranked in ChromaDB. Lexical ranking remains the default. |
| `embedding/embeddingModel.py` | Lazily loads the configured sentence-transformers model on CPU and caches its normalized embeddings. |
| `demo_data.py` | Validates MiniLM/Chroma configuration before allowing a live run. |
| `treeQA/wikidataUtills.py` | Truncates each Wikidata relation direction before the relation-selection prompt. |
| `eval/evaluate.py` | Adds SQuAD-style answer normalization, exact match, token F1, aliases, and containment reporting. |
| `eval_2wiki.py` | Adds seed-42 saved sampling, DR/no-correction/full TreeQA configurations, resumable JSONL, disk caches, Groq rate-limit backoff, confidence intervals, paired bootstrap summaries, and a prefix `--limit-questions` option. |
| `app.py` | Adds a 2Wiki comparative-results panel backed by the resumable JSONL, including partial progress, per-config metrics, paired differences, per-question traces, and JSON export; keeps single-question live inference and live QALD batch testing. The 2Wiki panel is scoped to the first three fixed-sample questions. |
| `.env.example` | Documents resource limits and opt-in MiniLM/Chroma configuration; lexical retrieval remains the default. |
| `tests/test_demo_data.py` | Adds re-verification, rewrite-cap, scoring, sampling, cache, retry, and harness tests. |
| `REPRODUCTION_NOTES.md` | Documents paper-vs-local setup, paper-reported scores, and limitations without presenting unrun results. |

## Tests and results

- `python -m unittest tests.test_demo_data -v` — **38 tests passed**.
- `python -m py_compile app.py treeQA_Config.py treeQA\tree_class\logicTree.py treeQA\wikipediaUtills.py treeQA\wikidataUtills.py embedding\embeddingModel.py demo_data.py eval_2wiki.py eval\evaluate.py tests\test_demo_data.py` — passed.
- Rendered app JavaScript syntax check with `node --check` — passed.
- `python eval_2wiki.py --help` — passed.
- The requested 20-question run was capped to the first **3** questions from
  its fixed seed-42 sample. All **9/9** DR, no-correction, and full TreeQA runs
  completed successfully and are visible in the app. This is a descriptive
  mini-pilot, not a statistically reliable benchmark result.
- Pilot Hits@1: DR **2/3 (66.7%)**, no-correction TreeQA **2/3 (66.7%)**, full
  TreeQA **3/3 (100%)**. Paired full TreeQA − DR difference: **+0.333**,
  bootstrap 95% CI **[0.000, 1.000]**; N=3 makes this highly uncertain.
- Mean tokens per question: DR **344**, no-correction TreeQA **6,990**, full
  TreeQA **23,347**. Mean time: **1.4 s**, **46.5 s**, and **144.5 s**,
  respectively. Full TreeQA used about **68x** DR tokens in this pilot.
  Current limits have not demonstrated token reduction relative to DR; no
  pre-change TreeQA baseline was collected.
- On question 1, full TreeQA corrected a wrong answer (no-correction EM 0,
  full TreeQA EM 1). Questions 2 and 3 were already correct under DR, so this
  tiny sample does not establish a general accuracy improvement.
- Several Groq 429 responses delayed the run; calls eventually completed.
- Fixed retrieval-retry instrumentation: repeated retrieval after a correction
  re-verification is no longer mislabeled as an unknown-evidence retry.
  Previously saved pilot rows retain their original retry counts; the pilot
  summary does not use that field.
- App API coverage verifies the 2Wiki summary endpoint and single-question
  live-inference response path using a mocked inference result; no additional
  billable live call was made during these checks.
- Browser verification on a fresh app server (`python app.py --port 8505`):
  Test Results showed all 9 2Wiki runs and the QALD test-run list; Refresh
  updated the panel and changing question selectors changed the displayed
  trace. Earlier Results rendered all 333 historical rows (320 successful,
  13 failed/missing). Live Inference rendered the manual-question textbox and
  run control. The UI check did not trigger an external model call.
- Offline CPU smoke test using the cached MiniLM model and temporary ChromaDB:
  the question ranked the “Capital” passage first (similarity 0.8512), ahead
  of unrelated text (-0.0352); querying with top-k 3 correctly returned only
  the two indexed passages. No Groq request or persistent project data was used.

## Not done, in priority order

1. For any performance claim, run a larger fixed paired sample when time/quota
   permit; do not present this N=3 pilot as conclusive.
2. Relik-Entity-Linking-Large is intentionally not installed or attempted; it
   differs from the paper setup.

## Commands

Run/resume the bounded three-question pilot:

```powershell
python eval_2wiki.py --run --n 20 --limit-questions 3 --seed 42
```

Print the Markdown summary:

```powershell
python eval_2wiki.py --summary result\2wiki_eval_seed42_n20.jsonl
```

Run the full 20-question comparison only if additional time/quota is
specifically approved:

```powershell
python eval_2wiki.py --run --n 20 --seed 42
```

The Groq key must be configured as `GROQ_API_KEY` in the ignored `.env` file.
Do not paste the key into source code or command history.

Optional local MiniLM retrieval (dependencies and model have been installed and
verified in the current environment):

```powershell
# In the existing ignored .env, set TREEQA_RETRIEVE_MODEL_NAME=minilm
# and TREEQA_CHROMA_STORE=true. Preserve the existing GROQ_API_KEY.
python -m unittest tests.test_demo_data -v
```

## Known bugs and limitations

- Tree principles such as MECE are prompt guidance, not mechanically validated.
- Entity linking remains LLM extraction/filtering plus Wikidata search, not
  Relik; relation linking is not evaluated with paper-style Recall@k.
- Wikipedia passages use lexical ranking by default. Optional MiniLM
  cosine-ranking is implemented and works with ChromaDB; it has not been
  evaluated against lexical ranking for QA quality.
- The 2Wiki script caches parsed Wikidata/Wikipedia retrieval outputs on disk.
  The three-question comparative sample completed all configurations, but its
  small size does not justify an aggregate performance claim.
- The final-answer LLM is instructed to emit a short answer, but output format
  compliance is not guaranteed by the provider.
- Free-tier rate limits can delay evaluation or produce recorded per-question
  failures; the runner does not bypass provider quotas.
- The web app displays the first three questions from the fixed seed-42, N=20
  output file. If another sample size or seed is evaluated, update the app's
  result-file selection before expecting that run to appear in the same panel.
- `requirements.txt` pins ChromaDB 0.6.3, while the verified environment has
  ChromaDB 1.5.9 and sentence-transformers 6.2.0. Both the focused tests and
  local smoke test should be run in the target environment before presentation.
- The MiniLM smoke test ran against ChromaDB 1.5.9; its `get(where=...,
  include=[])` API should also be checked against the repository-pinned 0.6.3
  environment before deployment.
