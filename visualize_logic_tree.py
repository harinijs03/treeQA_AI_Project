"""Render a stored TreeQA reasoning tree with Graphviz when available."""

import argparse
import itertools
import json
import os
import textwrap
from pathlib import Path

from demo_data import DEFAULT_RESULTS_PATH, load_jsonl, unpack_processed_answer


def format_label(text, max_width=50, indent=""):
    wrapped_lines = textwrap.wrap(str(text), width=max_width, subsequent_indent=indent + "  ")
    return indent + ("\n" + indent).join(wrapped_lines)


def _add_nodes_edges(graph, node_data, parent_id, id_counter):
    children = node_data.get("children", [])
    if not isinstance(children, list):
        return
    for child in children:
        if not isinstance(child, dict):
            continue
        current_id = f"node_{next(id_counter)}"
        label_parts = [
            f"Sub-Question:\n{format_label(child.get('sub_question', 'N/A'), indent='  ')}",
            f"\nHypothesis:\n{format_label(child.get('hypothesis_answer', 'N/A'), indent='  ')}",
        ]
        references = child.get("ref")
        if isinstance(references, dict):
            for source, source_items in references.items():
                items = source_items if isinstance(source_items, list) else [source_items]
                if items:
                    label_parts.append(f"\n{source.title()}:")
                    label_parts.extend(f"  - {format_label(item, indent='    ')}" for item in items)
        label = "\n".join(label_parts)
        graph.node(
            current_id,
            label=label + r"\l",
            shape="box",
            style="filled, rounded",
            fillcolor="lightblue",
        )
        graph.edge(parent_id, current_id)
        _add_nodes_edges(graph, child, current_id, id_counter)


def _tree_payload(data):
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON input: {error}") from error
    if not isinstance(data, dict):
        raise ValueError("Tree input must be a JSON object.")
    payload = data
    if "processed_answer" in payload:
        nested = unpack_processed_answer(payload)
        if nested is not None:
            payload = nested
    return payload


def visualize_logic_tree(
    json_data,
    output_filename="logic_tree_visualization",
    output_dir=".",
    view=False,
):
    """Render a tree object or serialized tree; return the rendered file path."""
    data = _tree_payload(json_data)
    try:
        from graphviz import Digraph, backend
    except ImportError as error:
        raise RuntimeError(
            "The Graphviz Python package is not installed. Install the project requirements "
            "and the Graphviz desktop executable, or use the Streamlit tree viewer."
        ) from error

    os.makedirs(output_dir, exist_ok=True)
    output_path_base = os.path.join(output_dir, output_filename)
    graph = Digraph(comment="TreeQA reasoning tree", format="png")
    graph.attr(rankdir="TB", overlap="false", splines="spline", nodesep="0.5", ranksep="0.8")
    graph.attr("node", shape="box", style="filled, rounded", fontname="Arial", fontsize="10")
    graph.attr("edge", fontname="Arial", fontsize="9")

    root_id = "node_0"
    root_label = (
        f"Input Question:\n{format_label(data.get('input_question', 'Question unavailable'))}"
        f"\n\nFinal Answer:\n{format_label(data.get('answer', 'Answer unavailable'))}"
    )
    graph.node(root_id, label=root_label + r"\l", fillcolor="lightcoral")
    tree = data.get("logic_tree")
    if isinstance(tree, dict):
        _add_nodes_edges(graph, tree, root_id, itertools.count(1))
    rendered_path = graph.render(output_path_base, view=view, cleanup=True)
    return rendered_path


def load_record(input_file, record_id=None, record_index=0):
    records, issues = load_jsonl(input_file)
    if record_id is not None:
        for record in records:
            if str(record.get("id", "")) == str(record_id):
                payload = unpack_processed_answer(record)
                if payload is None:
                    raise ValueError(f"Record {record_id!r} has no parseable reasoning tree.")
                return payload
        raise ValueError(f"Record ID {record_id!r} was not found in {input_file}.")
    if record_index < 0 or record_index >= len(records):
        raise ValueError(
            f"Record index {record_index} is out of range; {len(records)} valid record(s) loaded."
        )
    record = records[record_index]
    payload = unpack_processed_answer(record)
    if payload is None:
        raise ValueError(
            f"Record at index {record_index} has no parseable reasoning tree. "
            f"{len(issues)} malformed line(s) were skipped."
        )
    return payload


def main():
    parser = argparse.ArgumentParser(description="Render a TreeQA JSON or JSONL reasoning tree.")
    parser.add_argument(
        "--input-file",
        default=str(DEFAULT_RESULTS_PATH),
        help="JSON or JSONL file containing a TreeQA result (default: saved QALD results).",
    )
    parser.add_argument("--record-id", help="Select a JSONL record by its stored id.")
    parser.add_argument("--record-index", type=int, default=0, help="Zero-based JSONL record index.")
    parser.add_argument("--output", default="logic_tree_visualization", help="Output filename without extension.")
    parser.add_argument("--output-dir", default=".", help="Directory for the rendered image.")
    parser.add_argument("--view", action="store_true", help="Open the rendered image.")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if input_path.suffix.lower() == ".jsonl" or args.record_id is not None:
        data = load_record(input_path, args.record_id, args.record_index)
    else:
        data = json.loads(input_path.read_text(encoding="utf-8"))
    try:
        output = visualize_logic_tree(data, args.output, args.output_dir, args.view)
    except RuntimeError as error:
        parser.error(str(error))
    except Exception as error:
        parser.error(f"Could not render the reasoning tree: {error}")
    print(f"Graph saved to {output}")


if __name__ == "__main__":
    main()
