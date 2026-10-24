import json
from copy import deepcopy
import concurrent.futures
import threading
from queue import Queue


from LLMs.models import getModelResponse
from treeQA.getQueryInfo import getQueryInfo
from treeQA.tree_class.infoBox import infoBox
from treeQA_Config import (
    TREEQA_MAX_BRANCHING,
    TREEQA_MAX_RETRIEVAL_RETRIES,
    TREEQA_MAX_REWRITES_PER_NODE,
    TREEQA_MAX_TREE_DEPTH,
    TREEQA_MAX_TREE_UPDATES,
)


def limit_tree_size(root, max_depth=TREEQA_MAX_TREE_DEPTH, max_branching=TREEQA_MAX_BRANCHING):
    """Trim excess generated nodes to the configured depth and branching limits."""
    removed = 0

    def visit(node, depth):
        nonlocal removed
        children = node.get("children", [])
        if not isinstance(children, list):
            node["children"] = []
            return
        if len(children) > max_branching:
            removed += len(children) - max_branching
            children = children[:max_branching]
            node["children"] = children
        if depth >= max_depth:
            removed += len(children)
            node["children"] = []
            return
        for child in children:
            if isinstance(child, dict):
                visit(child, depth + 1)

    visit(root, 0)
    return removed



thread_local = threading.local()


class LogicTree:





    def __init__(self, data):






        self.data = data
        self.root = self.data.get("logic_tree", {})
        self.fix_count = 0
        self.tokenCount = 0
        self.trace_events = []
        self.node_rewrite_counts = {}
        self.tree_update_count = 0

        self.lock = threading.Lock()
        self.data["execution_trace"] = {
            "initial_tree": deepcopy(self.data.get("logic_tree", {})),
            "events": self.trace_events,
        }

    def record_trace_event(self, event):
        with self.lock:
            self.trace_events.append(event)

    @staticmethod
    def logic_tree_init(query):




        prompt = f"""You are an intelligent assistant who is good at analyzing and reasoning, and your task is to construct a logic tree to break down and reason step by step according to the complex questions posed by the user, and finally arrive at the answer.

    **Rules for constructing a logic tree:**

    - Maximum depth: {TREEQA_MAX_TREE_DEPTH} subquestion levels below the root.
    - Maximum branching factor: {TREEQA_MAX_BRANCHING} children per node.

    1. **Root:** the user's main Question.
    2. **Sub-nodes:** Each sub-node contains a **Sub-question** and a **Specific Hypothesis Answer** based on your own knowledge.
        * **Sub-questions:** are used to guide reasoning and information retrieval, and each sub-question should address only a single entity or relationship.
        * **Concrete Hypothesis Answer:** give the most likely concrete, verifiable answer in one short phrase or sentence.
    3. **Construction principles:** The construction of the logic tree follows the pyramid principle:
        * **Conclusion first:** The hypothetical answer of a parent node should be a generalization of the information of its children.
        * **Correspondence between top and bottom:** The information of a child node should support the hypothetical answer of the parent node.
        * **MECE Principle: ** Child nodes at the same level should be independent of each other and try to cover all possibilities.
        * **Logical progression:** The generation of child nodes should follow a certain logical order (e.g., time, cause and effect, etc.).

    * **Reasoning steps:**

    1. **Question Decomposition & Hypothesis Generation:** Based on the information (question or statement) in the current node, propose a simpler sub-question and give a **specific hypothetical answer** based on your own knowledge base.
    2. **Answer Synthesis:** When the logic tree is constructed, synthesize and reason out the final answer based on the information from all nodes.

    **Output Format:**

    Please strictly follow the following JSON format for output:

    ```json
    {{
      "input_question":"<main_question>",
      "logic_tree": {{
          "children": [
            {{
              "sub_question":"<subquestion1>",
              "hypothesis_answer":"<specific hypothesis answer 1>",
              "children": [
                {{
                  "sub_question":"<sub-sub-question 1.1>",
                  "hypothesis_answer": "<specific hypothetical answer 1.1>",
                  "children": [
                    {{
                      "sub_question":"<children_subquestion 1.1.1>",
                      "hypothesis_answer":"<specific hypothetical answer 1.1.1>"
                    }}
                  ]
                }}
              ]
            }}
          ]
        }},
      "answer": "<final answer>"
    }}
    ```

    Caveats:


    Each hypothetical answer must be a concrete, verifiable answer, not an abstract hypothesis.

    Ensure that the logic tree construction process follows the pyramid principle.

    The output should be concise and intuitive, with the lowest level question or hypothesis containing only the least granular information.

    Please just output json format content, do not output any analysis.

    Now the user question is:{query}
    """
        result, tokenCount = getModelResponse(prompt, query)
        result = result.replace("```json", "").replace("```", "")
        print("Tree construction complete!")

        try:
            json_string = result[result.find('{'):result.rfind('}') + 1]
            data = json.loads(json_string)
            data["input_question"] = query
            data["tree_limit_pruned_nodes"] = limit_tree_size(data.get("logic_tree", {}))
            return data, tokenCount
        except json.JSONDecodeError as e:
            print(f"Error: Decomposition failed, invalid JSON format. {e}")
            print("Received string:", result)
            return None, tokenCount

    def get_node_by_path(self, path):

        node = self.root
        try:
            for index in path:
                node = node["children"][index]
            return node
        except (KeyError, IndexError):
            return None

    def _increment_token_count(self, count):

        with self.lock:
            self.tokenCount += count

    def _increment_fix_count(self):

        with self.lock:
            self.fix_count += 1

    def _reserve_tree_update(self, path):
        path_key = tuple(path)
        with self.lock:
            rewrite_count = self.node_rewrite_counts.get(path_key, 0)
            if (
                rewrite_count >= TREEQA_MAX_REWRITES_PER_NODE
                or self.tree_update_count >= TREEQA_MAX_TREE_UPDATES
            ):
                return False
            self.node_rewrite_counts[path_key] = rewrite_count + 1
            self.tree_update_count += 1
            return True

    def _release_tree_update(self, path):
        path_key = tuple(path)
        with self.lock:
            self.node_rewrite_counts[path_key] -= 1
            self.tree_update_count -= 1

    def refine_subtree(self, path):





        thread_local.token_count = 0

        current_node = self.get_node_by_path(path)
        if not current_node:
            print(f"Warning: Node at path {path} not found, possibly due to a concurrent modification.")
            return

        question = self.data["input_question"]

        print(f"##################### Begin self-adaptive reasoning for path {path} #####################")
        checkInfoBox = infoBox()
        childQuestion = current_node.get("sub_question")
        hypothesis_answer = current_node.get("hypothesis_answer")

        getQueryInfo(childQuestion + hypothesis_answer, checkInfoBox, self, path=path)

        factCheckPrompt = f"""Please verify whether the answer is correct based on the given Info.  

    - If no relevant Info is provided (e.g., ["No Information provided."] or []), set `"isTrue": "unknown"` and `"fact_sufficient": false`.  
    - If the answer is correct and no reason is needed, set `"isTrue": true`.  
    - If the answer is incorrect, set `"isTrue": false`, provide the reason for the error, and please also suggest possible correct answers in the reason.  
    - Always include reference information (`ref`) when available, for both correct and incorrect answers.  

    ### Reference Formatting Rules:  
    - If no Info is available or unrelated to the question, set `"fact_sufficient": false`, `"ref": "No Information provided."`, and `"isTrue": "unknown"`.  
    - If citing `textInfo`, provide only the Wikipedia article's title ID (omit full text).  
    - If citing `graphInfo`, provide up to 3 relevant triplets in the format: `entityLabel-relationLabel-Value` from Wikidata.  
    - Do not fabricate information—references must match the provided Info.  

    ### Output Format (JSON only):  
    ```json
    {{
        "isTrue": true/false/unknown,
        "fact_sufficient": true/false,
        "reason": "<None>/<reason>",
        "ref": {{
            "wikipedia": ["<textInfo id>"],
            "wikidata": ["entityLabel-relationLabel-Value"]
        }}
    }}
    ```
            """
        query_info = f"question：{childQuestion}\nanswer:{hypothesis_answer}\nInfo：\t\ntextInfo:{checkInfoBox.textInfo}\t\ngraphInfo:{checkInfoBox.graphInfo}"
        result, tokenCount = getModelResponse(factCheckPrompt, query_info)
        result = result.replace('```json', '').replace('```', '')
        thread_local.token_count += tokenCount

        try:
            json_string = result[result.find('{'):result.rfind('}') + 1]
            resultJson = json.loads(json_string)
        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON from fact-check LLM for path {path}.")
            current_node["verification_status"] = "verification_error"
            self.record_trace_event({
                "type": "verification",
                "path": list(path),
                "status": "verification_error",
                "reason": "The model returned invalid verification JSON.",
            })
            self._increment_token_count(thread_local.token_count)
            return

        if (
            resultJson.get("isTrue") == "unknown"
            and not resultJson.get("fact_sufficient")
            and TREEQA_MAX_RETRIEVAL_RETRIES > 0
        ):
            prompt_new_cue = """The available information is insufficient to answer the question. Based on the given information and the question, generate a new clue to help retrieve the missing information needed to answer it.  
                ### Output Format (JSON only):  
                ```json
                {
                    "new_clue": "<New clue>"
                }"""
            new_clue_query = f"question：{childQuestion}\nCurrent Info：\t\ntextInfo:{checkInfoBox.textInfo}\t\ngraphInfo:{checkInfoBox.graphInfo}"
            new_clue_str, tokenCount = getModelResponse(prompt_new_cue, new_clue_query)
            thread_local.token_count += tokenCount
            print(f"######### No useful information for path {path}, new clue provided: {new_clue_str} #############")

            self.record_trace_event({
                "type": "verification",
                "path": list(path),
                "status": "unknown",
                "reason": resultJson.get("reason", "Evidence was insufficient."),
                "new_clue": new_clue_str,
            })
            self.record_trace_event({
                "type": "retrieval_retry",
                "path": list(path),
                "reason": "Verification returned unknown with insufficient evidence.",
            })
            getQueryInfo(new_clue_str, checkInfoBox, self, path=path)

            query_info_updated = f"question：{childQuestion}\nanswer:{hypothesis_answer}\nInfo：\t\ntextInfo:{checkInfoBox.textInfo}\t\ngraphInfo:{checkInfoBox.graphInfo}"
            result, tokenCount = getModelResponse(factCheckPrompt, query_info_updated)
            result = result.replace('```json', '').replace('```', '')
            thread_local.token_count += tokenCount
            try:
                json_string = result[result.find('{'):result.rfind('}') + 1]
                resultJson = json.loads(json_string)
            except json.JSONDecodeError:
                print(f"Error: Failed to decode JSON from second fact-check for path {path}.")
                current_node["verification_status"] = "verification_error"
                self.record_trace_event({
                    "type": "verification",
                    "path": list(path),
                    "status": "verification_error",
                    "reason": "The model returned invalid verification JSON after retrieval retry.",
                })
                self._increment_token_count(thread_local.token_count)
                return

        if resultJson.get("isTrue") is True:
            print(f"############ Evidence supports node at path {path}! ##############")
            current_node["verification_status"] = "verified"
            current_node["verification_reason"] = resultJson.get("reason")
            self.record_trace_event({
                "type": "verification",
                "path": list(path),
                "status": "verified",
                "reason": resultJson.get("reason", ""),
                "references": resultJson.get("ref", {}),
            })

            self._add_references_to_node(current_node, resultJson, checkInfoBox)

        elif resultJson.get("isTrue") is False:
            errorReason = resultJson.get("reason", "No reason provided.")
            print(f"############ Conflict found for node at path {path}: {errorReason} ###################")
            original_node = deepcopy(current_node)
            self.record_trace_event({
                "type": "verification",
                "path": list(path),
                "status": "contradicted",
                "reason": errorReason,
                "references": resultJson.get("ref", {}),
            })

            if not self._reserve_tree_update(path):
                current_node["verification_status"] = "correction_limit_reached"
                self.record_trace_event({
                    "type": "correction",
                    "path": list(path),
                    "reason": errorReason,
                    "status": "not_rewritten",
                    "error": "The per-node or per-question rewrite limit was reached.",
                    "before": deepcopy(current_node),
                    "after": deepcopy(current_node),
                })
                self._increment_token_count(thread_local.token_count)
                return

            fixHypothesisPropmt = f"""An error was found in the step {current_node} of the current assumption, with a specific error reason of {errorReason}.
                           Please fully review and refactor all sub_questions and hypothesis_answer starting at subtree node:{current_node}.
                           The refactoring process involves correcting or redoing each step as necessary based on the latest information and logic to ensure that the answer does not deviate from the current question:{question}. 
                   Keeping the original subtree depth and structure,you only need to fix the errors in the subtree node, and do not add any deeper child node in the subtree. 
                   Please just output json format content, do not output any analysis text.
                   """
            fix_query = f"Please be careful that current responses do not deviate from the question:{question}"
            result, tokenCount = getModelResponse(fixHypothesisPropmt, fix_query)
            result = result.replace('```json', '').replace('```', '')
            thread_local.token_count += tokenCount
            print(f"################ New subtree for path {path}: {result} ################")

            try:
                json_string = result[result.find('{'):result.rfind('}') + 1]
                fixedHypothesis = json.loads(json_string)
                if not isinstance(fixedHypothesis, dict):
                    raise ValueError("Corrected subtree must be a JSON object.")
                pruned_nodes = limit_tree_size(
                    fixedHypothesis,
                    max_depth=max(0, TREEQA_MAX_TREE_DEPTH - len(path)),
                )


                with self.lock:
                    current_node.clear()
                    current_node.update(fixedHypothesis)
                    current_node["verification_status"] = "corrected_pending_verification"
                    current_node["correction_reason"] = errorReason
                    self.fix_count += 1
                self.record_trace_event({
                    "type": "correction",
                    "path": list(path),
                    "reason": errorReason,
                    "status": "rewritten_pending_verification",
                    "rewrite_number": self.node_rewrite_counts[tuple(path)],
                    "tree_limit_pruned_nodes": pruned_nodes,
                    "before": original_node,
                    "after": deepcopy(current_node),
                })

                print(f"################ Subtree at path {path} update complete! ################")
                self._increment_token_count(thread_local.token_count)
                self.refine_subtree(path)
                rewritten_node = self.get_node_by_path(path)
                reverification_status = (
                    rewritten_node.get("verification_status")
                    if rewritten_node
                    else "node_missing"
                )
                self.record_trace_event({
                    "type": "correction_reverification",
                    "path": list(path),
                    "status": reverification_status,
                })
                return
            except (json.JSONDecodeError, ValueError):
                self._release_tree_update(path)
                print(f"Error: Failed to decode JSON for subtree fix at path {path}.")
                current_node["verification_status"] = "correction_error"
                self.record_trace_event({
                    "type": "correction",
                    "path": list(path),
                    "reason": errorReason,
                    "error": "The model returned invalid corrected-subtree JSON.",
                    "before": original_node,
                    "after": deepcopy(current_node),
                })
        else:
            print(f"Unable to check node at path {path}, continue to next node.")
            current_node["verification_status"] = "unknown"
            self.record_trace_event({
                "type": "verification",
                "path": list(path),
                "status": "unknown",
                "reason": resultJson.get("reason", "Evidence did not establish whether the hypothesis is correct."),
                "references": resultJson.get("ref", {}),
            })


        self._increment_token_count(thread_local.token_count)

    def _add_references_to_node(self, node, ref_json, info_box):



        with self.lock:
            if "ref" not in node:
                node["ref"] = {}


            wiki_ids = ref_json.get("ref", {}).get("wikipedia", [])


            wikipedia_ref_with_text = []
            if wiki_ids:
                for title_id in wiki_ids:
                    found = False
                    for item in info_box.textInfo:

                        if item[0]['id'] == title_id:
                            content = item[0]['content']
                            wikipedia_ref_with_text.append(f"{title_id}||{content}")
                            found = True
                            break
                    if not found:

                        wikipedia_ref_with_text.append(title_id)



            updated_ref = ref_json.get("ref", {}).copy()
            if wikipedia_ref_with_text:
                updated_ref["wikipedia"] = wikipedia_ref_with_text



            for key, value in updated_ref.items():
                if key not in node["ref"]:
                    node["ref"][key] = []

                for item in value:
                    if item not in node["ref"][key]:
                        node["ref"][key].append(item)
    def check_and_refine_parallel(self, max_workers=5):




        if not self.root.get("children"):
            print("Tree has no children to check.")
            return

        q = Queue()
        for i, child in enumerate(self.root.get("children", [])):
            q.put((child, [i]))

        level = 1
        while not q.empty():
            level_size = q.qsize()
            if level_size == 0:
                break

            print(f"\n--- Processing Level {level} with {level_size} nodes in parallel ---")

            tasks_for_current_level = []
            paths_for_next_level = []
            for _ in range(level_size):
                node, path = q.get()
                if "sub_question" in node and "hypothesis_answer" in node:
                    tasks_for_current_level.append(path)
                paths_for_next_level.append(path)

            if tasks_for_current_level:
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    executor.map(self.refine_subtree, tasks_for_current_level)

            print(f"--- Level {level} processing complete ---")


            for path in paths_for_next_level:
                parent_node = self.get_node_by_path(path)
                if parent_node and "children" in parent_node:
                    for i, child in enumerate(parent_node["children"]):
                        new_path = path + [i]
                        q.put((child, new_path))

            level += 1

    def check_and_refine_serial(self):





        def _recursive_check(node, path):
            if "sub_question" in node and "hypothesis_answer" in node:
                print(f"Checking (serial): {path}")
                self.refine_subtree(path)

            if "children" in node:

                current_node = self.get_node_by_path(path)
                if current_node and "children" in current_node:
                    for i, child in enumerate(current_node["children"]):
                        _recursive_check(child, path + [i])

        _recursive_check(self.root, [])

    def update_final_answer(self):
        """Generate a short answer from the refined reasoning tree."""
        info = self.data["logic_tree"]
        prompt = f"""
            Answer the question using the reasoning tree when helpful.
            Output only the shortest answer string that fully answers the question.
            Do not include an explanation, reasoning, markdown, or a leading phrase.
            question: {self.data["input_question"]}
            reasoning tree: {info}
         """
        final_answer, tokenCount = getModelResponse(prompt, self.data["input_question"])
        self.tokenCount += tokenCount
        print(f"######## The final answer is: ####################\n{final_answer}")
        self.data["answer"] = final_answer

    def to_json(self):

        return json.dumps(self.data, indent=4), self.fix_count

    def print_tree(self, node=None, indent=0, output_lines=None):

        if output_lines is None:
            output_lines = []

        if node is None:
            node = self.root
            output_lines.append(f"**Input Question:** {self.data.get('input_question', 'N/A')}\n")
            output_lines.append(f"**Final Answer:** {self.data.get('answer', 'N/A')}\n")

        if "sub_question" in node and "hypothesis_answer" in node:
            prefix = "  " * indent
            output_lines.append(f"{prefix}- Question: {node['sub_question']}")
            output_lines.append(f"{prefix}  Answer: {node['hypothesis_answer']}")
            if "ref" in node and node["ref"]:
                output_lines.append(f"{prefix}  References:")
                for ref_type, ref_list in node["ref"].items():
                    if ref_list:
                        for ref in ref_list:
                            output_lines.append(f"{prefix}    - {ref_type.capitalize()}: {ref}")

        if "children" in node:
            for child in node["children"]:
                self.print_tree(child, indent + 1, output_lines)

        return "\n".join(output_lines)


# ================== Example Usage ==================
if __name__ == '__main__':


    def getModelResponse(prompt, query):
        print("--- Mock LLM Call ---")
        print(f"Prompt: {prompt[:100]}...")
        print(f"Query: {query}")

        if "construct a logic tree" in prompt:
            mock_tree = {
                "input_question": "Who was the President of the United States when humans made their third landing on the moon?",
                "logic_tree": {
                    "children": [
                        {
                            "sub_question": "When was the third manned moon landing?",
                            "hypothesis_answer": "The third manned moon landing was Apollo 12 in November 1969.",
                            "children": [
                                {
                                    "sub_question": "Who was the US President in November 1969?",
                                    "hypothesis_answer": "Lyndon B. Johnson was the US President in November 1969."
                                }
                            ]
                        },
                        {
                            "sub_question": "Which mission was the third moon landing?",
                            "hypothesis_answer": "Apollo 14 was the third mission to land on the moon.",
                            "children": [
                                {
                                    "sub_question": "When did Apollo 14 land on the moon?",
                                    "hypothesis_answer": "Apollo 14 landed on February 5, 1971."
                                }
                            ]
                        }
                    ]
                },
                "answer": "Initial answer placeholder"
            }
            return json.dumps(mock_tree), 500
        elif "verify whether the answer is correct" in prompt:
            mock_verification = {"isTrue": False, "fact_sufficient": True,
                                 "reason": "The third landing was Apollo 14 in 1971, not Apollo 12. The president in 1971 was Richard Nixon, not LBJ.",
                                 "ref": {"wikipedia": ["Apollo_14"],
                                         "wikidata": ["Richard_Nixon-president_of-United_States"]}}
            return json.dumps(mock_verification), 100
        elif "refactor all sub_questions" in prompt:
            mock_fix = {
                "sub_question": "When was the third manned moon landing?",
                "hypothesis_answer": "The third manned moon landing was Apollo 14 on February 5, 1971.",
                "children": [
                    {
                        "sub_question": "Who was the US President on February 5, 1971?",
                        "hypothesis_answer": "Richard Nixon was the US President on February 5, 1971."
                    }
                ]
            }
            return json.dumps(mock_fix), 200
        elif "give a final, concise answer" in prompt:
            return "Richard Nixon.", 50
        return "{}", 10


    def getQueryInfo(query, infoBox, tree_instance):
        print(f"--- Mock Retrieval for query: '{query}' ---")
        infoBox.textInfo = [{"id": "Apollo_14",
                             "content": "Apollo 14 was the third United States Apollo mission to land on the Moon..."}]
        infoBox.graphInfo = ["Richard_Nixon-position_held-President_of_the_United_States"]


    class infoBox:
        def __init__(self):
            self.textInfo = []
            self.graphInfo = []



    user_query = "Who was the President of the United States when humans made their third landing on the moon?"


    initial_data, init_tokens = LogicTree.logic_tree_init(user_query)
    if initial_data:
        my_tree = LogicTree(initial_data)
        my_tree.tokenCount += init_tokens

        print("\n=============== INITIAL TREE ===============\n")
        print(my_tree.print_tree())


        print("\n=============== STARTING PARALLEL REFINEMENT ===============\n")
        my_tree.check_and_refine_parallel(max_workers=2)

        print("\n=============== REFINED TREE ===============\n")
        print(my_tree.print_tree())


        my_tree.update_final_answer()


        final_json, fix_count = my_tree.to_json()
        print("\n=============== FINAL JSON OUTPUT ===============\n")
        print(final_json)
        print(f"\nTotal fixes: {fix_count}")
        print(f"Total tokens used: {my_tree.tokenCount}")
