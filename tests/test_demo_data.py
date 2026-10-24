import json
import os
import re
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from io import StringIO
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen
from unittest.mock import patch

import app as demo_app
from app import DemoRequestHandler, configure_console_encoding
import LLMs.models as language_models
import treeQA_Config
from eval.evaluate import (
    evaluate_results,
    exact_match_score,
    normalize_answer,
    token_f1_score,
)
import eval_2wiki
from treeQA.tree_class.logicTree import LogicTree, limit_tree_size
from live_evaluation import (
    append_run_event,
    load_live_runs,
    select_benchmark_sample,
)
from demo_data import (
    DEFAULT_RESULTS_PATH,
    answer_matches,
    calculate_metrics,
    live_configuration_issues,
    load_jsonl,
    normalize_reasoning_tree,
    record_status,
)
from visualize_logic_tree import load_record


class ResultLoaderTests(unittest.TestCase):
    def test_loads_records_and_reports_malformed_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "results.jsonl"
            source.write_text(
                '{"id":"first","final_answer":"A"}\nnot json\n[]\n\n'
                '{"id":"second","final_answer":"B"}\n',
                encoding="utf-8",
            )
            records, issues = load_jsonl(source)
        self.assertEqual([record["id"] for record in records], ["first", "second"])
        self.assertEqual([record["_line_number"] for record in records], [1, 5])
        self.assertEqual([issue.line_number for issue in issues], [2, 3])

    def test_loads_all_saved_qald_records(self):
        records, issues = load_jsonl(DEFAULT_RESULTS_PATH)
        self.assertEqual(len(records), 333)
        self.assertEqual(issues, [])

    def test_live_sample_is_spread_across_benchmark_questions(self):
        sample = select_benchmark_sample(20)
        self.assertEqual(len(sample), 20)
        self.assertEqual(len({record["id"] for record in sample}), 20)
        self.assertEqual(sample[0]["id"], "qald_index_2")
        self.assertNotEqual(sample[-1]["id"], sample[0]["id"])

    def test_live_sample_rejects_counts_outside_the_requested_range(self):
        for count in (0, 26, True, "5"):
            with self.subTest(count=count), self.assertRaises(ValueError):
                select_benchmark_sample(count)
        self.assertEqual(len(select_benchmark_sample(5)), 5)

    def test_live_run_log_reconstructs_completed_run_and_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "runs.jsonl"
            append_run_event({
                "event": "run_started",
                "run_id": "run-one",
                "run_type": "sample",
                "started_at": "2026-10-23T00:00:00+00:00",
                "target_count": 1,
                "sample_ids": ["qald-1"],
            }, log_path)
            append_run_event({
                "event": "question_result",
                "run_id": "run-one",
                "record": {
                    "id": "qald-1",
                    "question": "Who?",
                    "original_answer": ["Ada"],
                    "final_answer": "Ada",
                    "fix_count": 1,
                    "total_processing_time": 12.5,
                    "total_tokens": 100,
                    "processed_answer": {
                        "answer": "Ada",
                        "execution_trace": {"events": [{"type": "verification"}]},
                    },
                    "run_index": 0,
                },
            }, log_path)
            append_run_event({
                "event": "run_finished",
                "run_id": "run-one",
                "status": "complete",
                "finished_at": "2026-10-23T00:01:00+00:00",
            }, log_path)
            runs, issues = load_live_runs(log_path)

        self.assertEqual(issues, [])
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["status"], "complete")
        self.assertEqual(runs[0]["completed_count"], 1)
        self.assertEqual(runs[0]["metrics"]["answer_match_rate"], 1.0)
        self.assertEqual(
            runs[0]["results"][0]["processed_answer"]["execution_trace"]["events"][0]["type"],
            "verification",
        )


class TreeNormalizationTests(unittest.TestCase):
    def test_normalizes_nested_tree_and_references(self):
        record = {
            "processed_answer": json.dumps({
                "logic_tree": {
                    "children": [{
                        "sub_question": "Who?",
                        "hypothesis_answer": "A person",
                        "ref": {"wikipedia": ["Example_0_0||Evidence"]},
                        "children": [{"sub_question": "When?", "children": []}],
                    }]
                }
            })
        }
        roots = normalize_reasoning_tree(record)
        self.assertEqual(len(roots), 1)
        self.assertEqual(roots[0].question, "Who?")
        self.assertEqual(roots[0].references["wikipedia"], ["Example_0_0||Evidence"])
        self.assertEqual(roots[0].children[0].question, "When?")

    def test_missing_or_invalid_tree_is_empty(self):
        self.assertEqual(normalize_reasoning_tree({}), ())
        self.assertEqual(normalize_reasoning_tree({"processed_answer": "not-json"}), ())

    def test_visualizer_loads_saved_jsonl_record_by_id(self):
        payload = load_record(DEFAULT_RESULTS_PATH, record_id="qald_index_2")
        self.assertIn("input_question", payload)
        self.assertIn("logic_tree", payload)


class MetricsTests(unittest.TestCase):
    def test_distinguishes_failed_missing_and_successful_records(self):
        records = [
            {
                "final_answer": "The answer is Alice.",
                "original_answer": ["Alice"],
                "total_processing_time": 12,
                "total_tokens": 100,
                "fix_count": 1,
            },
            {"final_answer": None, "fix_count": -1},
            {"final_answer": "", "fix_count": 0},
            {
                "final_answer": "Bob",
                "original_answer": ["Bob"],
                "total_processing_time": 20,
                "total_tokens": 200,
                "fix_count": 0,
            },
        ]
        metrics = calculate_metrics(records)
        self.assertEqual(
            [record_status(record) for record in records],
            ["Successful", "Failed", "Missing answer", "Successful"],
        )
        self.assertEqual(metrics["successful_count"], 2)
        self.assertEqual(metrics["failed_count"], 1)
        self.assertEqual(metrics["missing_answer_count"], 1)
        self.assertEqual(metrics["answer_match_rate"], 1.0)
        self.assertEqual(metrics["exact_match_rate"], 0.5)
        self.assertEqual(metrics["latency_mean"], 16)
        self.assertEqual(metrics["token_median"], 150)
        self.assertEqual(metrics["records_with_corrections"], 1)

    def test_matches_existing_containment_normalization(self):
        record = {"final_answer": "The answer is Alice.", "original_answer": ["Alice"]}
        self.assertTrue(answer_matches(record))
        self.assertFalse(answer_matches(record, exact=True))

    def test_saved_results_match_independent_legacy_score(self):
        records, _ = load_jsonl(DEFAULT_RESULTS_PATH)
        metrics = calculate_metrics(records)
        self.assertEqual(metrics["record_count"], 333)
        self.assertEqual(metrics["successful_count"], 320)
        self.assertEqual(metrics["failed_count"], 13)
        self.assertEqual(metrics["answer_match_count"], 213)
        self.assertAlmostEqual(metrics["answer_match_rate"], 213 / 320)


class AnswerScoringTests(unittest.TestCase):
    def test_squad_normalization_and_token_f1(self):
        self.assertEqual(normalize_answer("The Eiffel Tower!"), "eiffel tower")
        self.assertEqual(exact_match_score("Thomas Newman", ["Thomas Montgomery Newman"]), 0.0)
        self.assertAlmostEqual(
            token_f1_score("Thomas Newman", ["Thomas Montgomery Newman"]),
            0.8,
        )

    def test_evaluator_prefers_short_answer_and_includes_record_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "answers.jsonl"
            source.write_text(
                json.dumps({
                    "id": "q1",
                    "question": "Who composed it?",
                    "original_answer": ["Thomas Montgomery Newman"],
                    "answer_aliases": ["Thomas Newman"],
                    "final_answer": "A long explanation about Thomas Newman.",
                    "short_answer": "Thomas Newman",
                }) + "\n",
                encoding="utf-8",
            )
            output = StringIO()
            with redirect_stdout(output):
                evaluate_results(str(source))
        self.assertIn("Hits@1 (normalized exact match): 1.0000", output.getvalue())
        self.assertIn("Token-level F1:                  1.0000", output.getvalue())


class TwoWikiHarnessTests(unittest.TestCase):
    def test_sample_is_fixed_and_contains_gold_evidence(self):
        rows = [
            {
                "_id": f"q{index}",
                "question": f"Question {index}",
                "answer": f"Answer {index}",
                "supporting_facts": [["Title", 0]],
                "evidences": [["Head", "relation", "Tail"]],
            }
            for index in range(6)
        ]
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset.json"
            dataset.write_text(json.dumps(rows), encoding="utf-8")
            with (
                patch.object(eval_2wiki, "DATASET_PATH", dataset),
                patch.object(eval_2wiki, "RESULTS_DIR", Path(directory) / "results"),
            ):
                first = eval_2wiki.load_sample(3, seed=42)
                second = eval_2wiki.load_sample(3, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertIn("supporting_facts", first[0])
        self.assertIn("evidences_id", first[0])

    def test_run_evaluation_limits_work_to_prefix_of_saved_sample(self):
        sample = [
            {"id": f"q{index}", "question": f"Question {index}", "gold": "A"}
            for index in range(4)
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.jsonl"
            with (
                patch.object(eval_2wiki.treeQA_Config, "model_name", "groq"),
                patch.object(eval_2wiki.treeQA_Config, "groqApiKey", "gsk-test"),
                patch.object(eval_2wiki, "load_sample", return_value=sample),
                patch.object(eval_2wiki, "RESULTS_DIR", Path(directory)),
                patch.object(
                    eval_2wiki,
                    "run_one",
                    side_effect=lambda row, config, _cache: {
                        "id": row["id"],
                        "config": config,
                        "error": None,
                        "hits_at_1": 1.0,
                        "f1": 1.0,
                        "containment": 1.0,
                        "total_tokens": 1,
                        "wall_time_seconds": 1.0,
                        "tree_updates": 0,
                    },
                ) as run_one,
            ):
                result_path = eval_2wiki.run_evaluation(
                    4,
                    seed=42,
                    output_path=output,
                    limit_questions=2,
                )
                self.assertEqual(result_path, output)
                self.assertEqual(run_one.call_count, 6)
                self.assertEqual(
                    {json.loads(line)["id"] for line in output.read_text(encoding="utf-8").splitlines()},
                    {"q0", "q1"},
                )

    def test_rate_limit_retry_after_and_paired_summary_helpers(self):
        error = RuntimeError("429 rate limit")
        error.response = SimpleNamespace(headers={"retry-after": "7"})
        self.assertEqual(eval_2wiki.retry_after_seconds(error, 0), 7.0)
        self.assertEqual(eval_2wiki.wilson_interval(0, 1)[0], 0.0)
        mean, low, high = eval_2wiki.paired_bootstrap_difference([
            {"id": "q1", "config": "dr", "hits_at_1": 0},
            {"id": "q1", "config": "treeqa_full", "hits_at_1": 1},
        ], "hits_at_1", iterations=100)
        self.assertEqual(mean, 1.0)
        self.assertEqual((low, high), (1.0, 1.0))

    def test_retrieval_retry_count_excludes_rewrite_reverification_retrievals(self):
        events = [
            {"type": "retrieval", "path": [0]},
            {"type": "retrieval", "path": [1]},
            {"type": "retrieval", "path": [1]},
            {"type": "retrieval_retry", "path": [1]},
            {"type": "retrieval", "path": [1]},
        ]
        self.assertEqual(eval_2wiki.count_retrieval_retries(events), 1)

    def test_disk_cache_reuses_wikidata_json_response(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = {"search": [{"id": "Q1"}]}
            with patch.object(eval_2wiki.wikidata, "safe_request", return_value=payload) as fetch:
                originals = eval_2wiki.install_disk_caches(Path(directory))
                try:
                    first = eval_2wiki.wikidata.safe_request("https://example.test", {"q": "x"}, None)
                    second = eval_2wiki.wikidata.safe_request("https://example.test", {"q": "x"}, None)
                finally:
                    eval_2wiki.restore_disk_caches(originals)
        self.assertEqual(first, payload)
        self.assertEqual(second, payload)
        fetch.assert_called_once()

    def test_failed_records_can_resume_and_summary_uses_latest_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            rows = [
                {"id": "q1", "config": "dr", "error": "429", "hits_at_1": 0},
                {"id": "q1", "config": "dr", "error": None, "hits_at_1": 1},
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            self.assertEqual(eval_2wiki.load_completed(path), {("q1", "dr")})
            latest = eval_2wiki.read_results(path)
        self.assertEqual(len(latest), 1)
        self.assertEqual(latest[0]["hits_at_1"], 1)

    def test_direct_baseline_records_short_answer_and_usage(self):
        sample = {
            "id": "q1",
            "question": "Who?",
            "gold": "Alice",
            "answer_aliases": [],
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(eval_2wiki.models, "getModelResponse", return_value=("Alice", 14)),
        ):
            record = eval_2wiki.run_one(sample, "dr", Path(directory))
        self.assertEqual(record["short_answer"], "Alice")
        self.assertEqual(record["total_tokens"], 14)
        self.assertEqual(record["hits_at_1"], 1.0)
        self.assertIsNone(record["error"])


class LogicTreeCorrectionTests(unittest.TestCase):
    def test_tree_limits_prune_depth_and_branching(self):
        tree = {
            "children": [
                {"children": [{"children": []}, {"children": []}]},
                {"children": []},
                {"children": []},
            ]
        }
        removed = limit_tree_size(tree, max_depth=1, max_branching=2)
        self.assertEqual(removed, 3)
        self.assertEqual(len(tree["children"]), 2)
        self.assertEqual(tree["children"][0]["children"], [])

    def test_rewritten_node_is_retrieved_and_reverified_before_return(self):
        tree_data = {
            "input_question": "Where was the person born?",
            "logic_tree": {
                "children": [{
                    "sub_question": "Where was Alex born?",
                    "hypothesis_answer": "Paris",
                    "children": [],
                }]
            },
        }
        contradiction = json.dumps({
            "isTrue": False,
            "fact_sufficient": True,
            "reason": "The retrieved evidence says Lyon.",
            "ref": {},
        })
        corrected_subtree = json.dumps({
            "sub_question": "Where was Alex born?",
            "hypothesis_answer": "Lyon",
            "children": [],
        })
        supported = json.dumps({
            "isTrue": True,
            "fact_sufficient": True,
            "reason": "The retrieved evidence supports Lyon.",
            "ref": {},
        })

        with (
            patch("treeQA.tree_class.logicTree.getQueryInfo") as retrieve,
            patch(
                "treeQA.tree_class.logicTree.getModelResponse",
                side_effect=[
                    (contradiction, 10),
                    (corrected_subtree, 20),
                    (supported, 10),
                ],
            ) as model_response,
        ):
            tree = LogicTree(tree_data)
            tree.refine_subtree([0])

        node = tree.get_node_by_path([0])
        self.assertEqual(node["hypothesis_answer"], "Lyon")
        self.assertEqual(node["verification_status"], "verified")
        self.assertEqual(retrieve.call_count, 2)
        self.assertEqual(model_response.call_count, 3)
        self.assertEqual(tree.fix_count, 1)
        self.assertEqual(tree.tree_update_count, 1)
        self.assertEqual(
            [
                (event["type"], event.get("status"))
                for event in tree.trace_events
                if event["type"] in {"correction", "correction_reverification"}
            ],
            [
                ("correction", "rewritten_pending_verification"),
                ("correction_reverification", "verified"),
            ],
        )

    def test_node_rewrite_limit_stops_a_second_correction_attempt(self):
        tree = LogicTree({
            "input_question": "Where was the person born?",
            "logic_tree": {
                "children": [{
                    "sub_question": "Where was Alex born?",
                    "hypothesis_answer": "Paris",
                    "children": [],
                }]
            },
        })
        contradiction = json.dumps({
            "isTrue": False,
            "fact_sufficient": True,
            "reason": "Evidence contradicts the hypothesis.",
            "ref": {},
        })

        with (
            patch("treeQA.tree_class.logicTree.getQueryInfo"),
            patch(
                "treeQA.tree_class.logicTree.getModelResponse",
                return_value=(contradiction, 10),
            ) as model_response,
        ):
            tree.node_rewrite_counts[(0,)] = 1
            tree.tree_update_count = 1
            tree.refine_subtree([0])

        self.assertEqual(
            tree.get_node_by_path([0])["verification_status"],
            "correction_limit_reached",
        )
        self.assertEqual(model_response.call_count, 1)
        self.assertEqual(tree.tree_update_count, 1)
        self.assertEqual(tree.fix_count, 0)


class LiveInferenceConfigurationTests(unittest.TestCase):
    def test_env_loader_reads_values_without_dotenv_and_preserves_process_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text(
                'GROQ_API_KEY="local-test-key"\n'
                "TREEQA_MODEL_NAME=groq # comment\n"
                "TREEQA_PROXY=\n",
                encoding="utf-8",
            )
            with (
                patch.object(treeQA_Config, "load_dotenv", None),
                patch.dict(
                    "os.environ",
                    {"GROQ_API_KEY": "process-key"},
                    clear=True,
                ),
            ):
                treeQA_Config._load_env_file(env_file)
                self.assertEqual(
                    os.environ["GROQ_API_KEY"],
                    "process-key",
                )
                self.assertEqual(os.environ["TREEQA_MODEL_NAME"], "groq")
                self.assertEqual(os.environ["TREEQA_PROXY"], "")

    def test_groq_only_configuration_needs_no_other_provider_credentials(self):
        import treeQA_Config

        with patch.multiple(
            treeQA_Config,
            model_name="groq",
            groqApiKey="gsk-test-key",
            el_model="wikidata",
            RetrieveModelName="lexical",
            Chroma_store=False,
        ):
            self.assertEqual(live_configuration_issues(), [])

    def test_groq_configuration_reports_missing_key(self):
        import treeQA_Config

        with patch.multiple(
            treeQA_Config,
            model_name="groq",
            groqApiKey="",
            el_model="wikidata",
            RetrieveModelName="lexical",
            Chroma_store=False,
        ):
            self.assertEqual(
                live_configuration_issues(),
                ["Set a valid GROQ_API_KEY for model groq."],
            )

    def test_minilm_configuration_is_accepted_when_dependencies_and_chroma_are_enabled(self):
        with patch.multiple(
            treeQA_Config,
            model_name="groq",
            groqApiKey="gsk-test-key",
            el_model="wikidata",
            RetrieveModelName="minilm",
            TREEQA_EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2",
            Chroma_store=True,
        ):
            self.assertEqual(live_configuration_issues(), [])

    def test_minilm_configuration_requires_chroma(self):
        with patch.multiple(
            treeQA_Config,
            model_name="groq",
            groqApiKey="gsk-test-key",
            el_model="wikidata",
            RetrieveModelName="minilm",
            Chroma_store=False,
        ):
            self.assertEqual(
                live_configuration_issues(),
                ["Set TREEQA_CHROMA_STORE=true to enable MiniLM dense retrieval."],
            )

    def test_groq_client_uses_configured_model_and_openai_compatible_endpoint(self):
        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))],
            usage=SimpleNamespace(total_tokens=12),
        )
        with (
            patch.object(language_models, "groqApiKey", "gsk-test-key"),
            patch.object(language_models, "groqModel", "llama-test"),
            patch.object(language_models, "OpenAI") as openai,
        ):
            openai.return_value.chat.completions.create.return_value = completion
            response, tokens = language_models.get_groq_response("system", "question")

        self.assertEqual((response, tokens), ("answer", 12))
        openai.assert_called_once_with(
            api_key="gsk-test-key",
            base_url="https://api.groq.com/openai/v1",
            max_retries=5,
        )
        self.assertEqual(
            openai.return_value.chat.completions.create.call_args.kwargs["model"],
            "llama-test",
        )


class LexicalWikipediaRetrievalTests(unittest.TestCase):
    def test_direct_retrieval_ranks_matching_wikipedia_passage_without_embeddings(self):
        from treeQA import wikipediaUtills

        sections = [
            {"title": "Overview", "content": "This section discusses unrelated history."},
            {
                "title": "Capital",
                "content": "Wellington is the capital city of New Zealand.",
            },
        ]
        with patch.object(wikipediaUtills, "get_article_sections", return_value=sections):
            results = wikipediaUtills.getWikipediaResultDirect(
                "New Zealand", "What is the capital of New Zealand?", top_k=1
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Capital")
        self.assertIn("Wellington", results[0]["content"])

    def test_chroma_retrieval_returns_nearest_cosine_matches_in_rank_order(self):
        from treeQA import wikipediaUtills

        collection = SimpleNamespace(
            get=unittest.mock.Mock(return_value={"ids": ["chunk-1", "chunk-2"]}),
            query=unittest.mock.Mock(return_value={
                "ids": [["poor-match", "best-match"]],
                "metadatas": [[
                    {"title": "Other", "content": "Unrelated"},
                    {"title": "Answer", "content": "Wellington"},
                ]],
                "distances": [[0.8, 0.1]],
            })
        )
        with (
            patch.object(wikipediaUtills, "_get_collection", return_value=collection),
            patch("embedding.embeddingModel.getEmbeddings", return_value=[[0.1, 0.2]]),
        ):
            results = wikipediaUtills.query_article("capital?", 2, "New Zealand")

        self.assertEqual([result["id"] for result in results], ["poor-match", "best-match"])
        self.assertAlmostEqual(results[0]["similarity"], 0.2)
        self.assertAlmostEqual(results[1]["similarity"], 0.9)
        self.assertEqual(collection.query.call_args.kwargs["where"], {
            "article_title": {"$eq": "New Zealand"}
        })
        self.assertEqual(collection.query.call_args.kwargs["n_results"], 2)
        self.assertEqual(collection.get.call_args.kwargs["include"], [])

    def test_chroma_retrieval_returns_empty_for_articles_without_indexed_chunks(self):
        from treeQA import wikipediaUtills

        collection = SimpleNamespace(
            get=unittest.mock.Mock(return_value={"ids": []}),
            query=unittest.mock.Mock(),
        )
        with (
            patch.object(wikipediaUtills, "_get_collection", return_value=collection),
            patch("embedding.embeddingModel.getEmbeddings", return_value=[[0.1, 0.2]]),
        ):
            results = wikipediaUtills.query_article("capital?", 3, "New Zealand")

        self.assertEqual(results, [])
        collection.query.assert_not_called()

    def test_minilm_encoder_is_lazy_cpu_normalized_and_cached(self):
        from embedding import embeddingModel

        encoder = SimpleNamespace(
            encode=unittest.mock.Mock(return_value=__import__("numpy").array([[0.6, 0.8]]))
        )
        with (
            patch.object(embeddingModel, "_sentence_transformer", None),
            patch("sentence_transformers.SentenceTransformer", return_value=encoder) as factory,
            patch.object(embeddingModel, "RetrieveModelName", "minilm"),
        ):
            first = embeddingModel.getEmbeddings(["text"])
            second = embeddingModel.getEmbeddings(["next"])

        self.assertEqual(first, [[0.6, 0.8]])
        self.assertEqual(second, [[0.6, 0.8]])
        factory.assert_called_once_with(
            treeQA_Config.TREEQA_EMBEDDING_MODEL,
            device="cpu",
        )
        self.assertTrue(encoder.encode.call_args.kwargs["normalize_embeddings"])


class OfflineWebAppTests(unittest.TestCase):
    def test_console_encoding_uses_utf8_with_safe_error_handling(self):
        class TestStream:
            def __init__(self):
                self.options = None

            def reconfigure(self, **options):
                self.options = options

        stdout = TestStream()
        stderr = TestStream()
        configure_console_encoding(stdout, stderr)
        self.assertEqual(
            stdout.options,
            {"encoding": "utf-8", "errors": "backslashreplace"},
        )
        self.assertEqual(stderr.options, stdout.options)

    def test_home_page_shows_live_and_earlier_results_separately(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), DemoRequestHandler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            with urlopen(f"http://127.0.0.1:{server.server_port}/", timeout=10) as response:
                body = response.read().decode("utf-8")
                self.assertEqual(response.status, 200)
            self.assertNotIn('data-tab="saved"', body)
            self.assertNotIn('data-tab="evaluation"', body)
            self.assertNotIn('data-tab="pipeline"', body)
            self.assertIn("qald_index_2", body)
            self.assertNotIn("ds3_qald.jsonl", body)
            self.assertIn(">Test Results</button>", body)
            self.assertIn(">Earlier Results</button>", body)
            self.assertIn("Previously recorded benchmark outputs", body)
            self.assertIn("Live Inference", body)
            self.assertIn("2Wiki comparative pilot", body)
            self.assertIn("Per-question trace", body)
            self.assertIn("/api/2wiki-evaluation", body)
            self.assertIn(
                'Questions per run ("+String(data.min_questions||1)+"-"+String(data.max_questions||25)+")',
                body,
            )
            self.assertIn("Hits@1 (normalized exact)", body)
            self.assertIn("Precision@3 is not available", body)
            for hidden_copy in (
                "Saved Results Demo",
                "Evaluation Dashboard",
                "System Pipeline",
                "Run measurements are historical",
                "GROQ_API_KEY locally",
                "Calls Groq and public knowledge services only",
                "This is a real external-service run",
            ):
                self.assertNotIn(hidden_copy, body.split('<script id="earlier-records-data"', 1)[0])
            earlier_json = re.search(
                r'<script id="earlier-records-data" type="application/json">(.*?)</script>',
                body,
                re.DOTALL,
            )
            self.assertIsNotNone(earlier_json)
            earlier_records = json.loads(earlier_json.group(1))
            failed = next(record for record in earlier_records if record["status"] == "Failed")
            self.assertFalse(failed["containment_match"])
            self.assertFalse(failed["exact_match"])
            self.assertNotIn("inference", sys.modules)
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=5)

    def test_2wiki_evaluation_api_summarizes_latest_successful_records(self):
        records = [
            {
                "id": "q1",
                "config": "dr",
                "question": "Question?",
                "gold": "Answer",
                "short_answer": "Answer",
                "hits_at_1": 1.0,
                "f1": 1.0,
                "containment": 1.0,
                "total_tokens": 10,
                "wall_time_seconds": 1.0,
                "tree_updates": 0,
                "error": None,
            },
            {
                "id": "q1",
                "config": "treeqa_full",
                "question": "Question?",
                "gold": "Answer",
                "short_answer": "",
                "hits_at_1": 0.0,
                "f1": 0.0,
                "containment": 0.0,
                "total_tokens": 20,
                "wall_time_seconds": 2.0,
                "tree_updates": 1,
                "error": "Rate limit",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            results_path = Path(directory) / "results.jsonl"
            sample_path = Path(directory) / "sample.json"
            results_path.write_text(
                "".join(json.dumps(row) + "\n" for row in records),
                encoding="utf-8",
            )
            sample_path.write_text(
                json.dumps([{"id": "q1"}, {"id": "q2"}]),
                encoding="utf-8",
            )
            with (
                patch.object(demo_app, "TWO_WIKI_RESULTS_PATH", results_path),
                patch.object(demo_app, "TWO_WIKI_SAMPLE_PATH", sample_path),
                patch.object(demo_app, "TWO_WIKI_DISPLAY_QUESTION_LIMIT", 1),
            ):
                payload = demo_app.load_2wiki_evaluation()

        self.assertEqual(payload["sample_size"], 1)
        self.assertEqual(payload["completed_tasks"], 1)
        self.assertEqual(payload["failed_attempts"], 1)
        self.assertEqual(payload["summary"]["dr"]["n"], 1)
        self.assertEqual(payload["summary"]["treeqa_full"]["n"], 0)
        self.assertEqual(len(payload["records"]), 2)

    def test_single_live_inference_api_returns_answer_and_metrics(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), DemoRequestHandler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        record = {
            "final_answer": "Ada Lovelace",
            "fix_count": 0,
            "total_processing_time": 1.5,
            "total_tokens": 50,
            "processed_answer": {"answer": "Ada Lovelace", "logic_tree": {"children": []}},
        }
        try:
            with (
                patch.object(demo_app, "live_configuration_issues", return_value=[]),
                patch.object(demo_app, "append_run_event"),
                patch.object(demo_app, "_execute_live_question", return_value=record),
            ):
                request = Request(
                    f"http://127.0.0.1:{server.server_port}/api/live",
                    data=json.dumps({"question": "Who was Ada Lovelace?"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(request, timeout=5) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                    self.assertEqual(response.status, 200)
            self.assertTrue(payload["success"])
            self.assertEqual(payload["metrics"]["final_answer"], "Ada Lovelace")
            self.assertEqual(payload["metrics"]["total_tokens"], 50)
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=5)

    def test_live_evaluation_api_starts_batch_without_single_question_field(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), DemoRequestHandler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        finished = threading.Event()
        sample = [
            {"id": f"qald-{index}", "question": f"Who {index}?", "original_answer": ["Ada"]}
            for index in range(20)
        ]
        try:
            with (
                patch.object(demo_app, "live_configuration_issues", return_value=[]),
                patch.object(demo_app, "select_benchmark_sample", return_value=sample),
                patch.object(demo_app, "append_run_event"),
                patch.object(
                    demo_app,
                    "_run_live_sample",
                    side_effect=lambda *_args: finished.set(),
                ),
                patch.object(demo_app, "active_live_run", None),
            ):
                request = Request(
                    f"http://127.0.0.1:{server.server_port}/api/live-evaluations",
                    data=json.dumps({"count": 20}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(request, timeout=5) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                    self.assertEqual(response.status, 202)
                self.assertTrue(payload["success"])
                self.assertEqual(payload["target_count"], 20)
                self.assertTrue(finished.wait(timeout=2))
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
