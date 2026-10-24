import argparse
import json
import os
import logging
import re
import string
from collections import Counter
from typing import List, Dict, Any, Optional


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def normalize_text(s: str) -> str:

    if not s:
        return ""
    s = str(s).lower()
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ' '.join(s.split())
    return s


def normalize_answer(s: str) -> str:
    """Apply SQuAD-style lowercase, punctuation, article, and whitespace normalization."""
    text = str(s or "").lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match_score(prediction: str, ground_truths: List[str]) -> float:
    normalized_prediction = normalize_answer(prediction)
    return float(any(
        normalized_prediction == normalize_answer(answer)
        for answer in ground_truths
    ))


def token_f1_score(prediction: str, ground_truths: List[str]) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    best_f1 = 0.0
    for answer in ground_truths:
        answer_tokens = normalize_answer(answer).split()
        if not prediction_tokens or not answer_tokens:
            score = float(prediction_tokens == answer_tokens)
        else:
            common = sum((Counter(prediction_tokens) & Counter(answer_tokens)).values())
            if common == 0:
                score = 0.0
            else:
                precision = common / len(prediction_tokens)
                recall = common / len(answer_tokens)
                score = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, score)
    return best_f1


def containment_score(prediction: str, ground_truths: List[str]) -> float:
    normalized_prediction = normalize_answer(prediction)
    return float(any(
        normalize_answer(answer) in normalized_prediction
        for answer in ground_truths
        if normalize_answer(answer)
    ))


def expand_answers(
    answers: List[str],
    alias_data: Optional[Dict[str, List[str]]] = None,
    record_aliases: Optional[List[str]] = None,
) -> List[str]:
    expanded = list(answers)
    expanded.extend(str(alias) for alias in (record_aliases or []) if alias is not None)
    if alias_data:
        for answer in answers:
            aliases = alias_data.get(answer, [])
            if isinstance(aliases, str):
                aliases = [aliases]
            if isinstance(aliases, list):
                expanded.extend(str(alias) for alias in aliases if alias is not None)
    return list(dict.fromkeys(expanded))


def calculate_em_contains(prediction: str, ground_truths: List[str]) -> bool:




    if not ground_truths:
        return not normalize_text(prediction)
    if not prediction:
        return False

    normalized_prediction = normalize_text(prediction)
    if not normalized_prediction:
        return False

    for gt in ground_truths:
        normalized_gt = normalize_text(gt)
        if not normalized_gt:
            continue
        if normalized_gt in normalized_prediction:
            return True
    return False

def load_aliases(alias_file_path: str) -> Optional[Dict[str, List[str]]]:


    if not alias_file_path or not os.path.isfile(alias_file_path):
        logging.error(f"提供的别名文件路径无效或不存在: {alias_file_path}")
        return None
    try:
        with open(alias_file_path, 'r', encoding='utf-8') as f:
            aliases = json.load(f)

        if not isinstance(aliases, dict):
             logging.error(f"别名文件顶层结构不是字典: {alias_file_path}")
             return None
        logging.info(f"成功从 {alias_file_path} 加载 {len(aliases)} 条别名记录")
        return aliases
    except json.JSONDecodeError as e:
        logging.error(f"解析别名文件 JSON 时出错: {alias_file_path} - {e}")
        return None
    except Exception as e:
        logging.error(f"加载别名文件时发生意外错误: {alias_file_path} - {e}")
        return None


def evaluate_results(input_file_path: str,
                     alias_data: Optional[Dict[str, List[str]]] = None,
                     error_file_path: str = None):








    if not os.path.exists(input_file_path):
        logging.error(f"输入文件未找到: {input_file_path}")
        return

    total_records = 0
    em_scores = []
    f1_scores = []
    containment_scores = []

    metric_accumulators: Dict[str, float] = {
        "logic_init_time": 0.0, "self_adaptive_time": 0.0, "final_reasoning_time": 0.0,
        "total_processing_time": 0.0, "logic_init_tokens": 0.0, "self_adaptive_tokens": 0.0,
        "final_reasoning_tokens": 0.0, "total_tokens": 0.0, "fix_count": 0.0,
    }
    metric_keys = list(metric_accumulators.keys())
    error_records = []

    logging.info(f"开始评估文件: {input_file_path}")
    if alias_data:
        logging.info("检测到已加载的别名数据，将使用别名进行 EM 评估。")
    else:
        logging.info("未提供或加载别名数据失败，EM 评估将仅使用原始答案。")

    with open(input_file_path, 'r', encoding='utf-8') as infile:
        for i, line in enumerate(infile):
            try:
                data = json.loads(line)
                total_records += 1

                record_id = data.get("id", f"line_{i+1}")
                question = data.get("question", "")
                original_answers_raw = data.get("original_answer", [])
                if original_answers_raw is None: original_answers = []
                elif isinstance(original_answers_raw, list): original_answers = [str(ans) for ans in original_answers_raw if ans is not None]
                else: original_answers = [str(original_answers_raw)]


                answers_to_match = list(original_answers)
                if alias_data:
                    extended_answers = list(original_answers)
                    for ans in original_answers:
                        aliases = alias_data.get(ans, [])
                        if aliases:
                            extended_answers.extend(aliases)
                    answers_to_match = list(set(extended_answers))


                record_aliases = data.get("answer_aliases", [])
                if isinstance(record_aliases, str):
                    record_aliases = [record_aliases]
                if isinstance(record_aliases, list):
                    answers_to_match.extend(str(alias) for alias in record_aliases if alias is not None)
                    answers_to_match = list(dict.fromkeys(answers_to_match))

                prediction = data.get("short_answer") or data.get("final_answer")
                if prediction is None:
                    processed_answer_str = data.get("processed_answer")
                    if isinstance(processed_answer_str, str):
                        try:
                            processed_answer_json = json.loads(processed_answer_str)
                            prediction = processed_answer_json.get("answer")
                        except json.JSONDecodeError: prediction = None
                    elif isinstance(processed_answer_str, dict): prediction = processed_answer_str.get("answer")

                if prediction is None: prediction = ""
                else: prediction = str(prediction)


                em = exact_match_score(prediction, answers_to_match)
                em_scores.append(em)
                f1_scores.append(token_f1_score(prediction, answers_to_match))
                containment_scores.append(containment_score(prediction, answers_to_match))


                for key in metric_keys:
                    value = data.get(key, 0)
                    if isinstance(value, (int, float)): metric_accumulators[key] += value
                    else: logging.warning(f"记录 {record_id} 指标 '{key}' 非数值: {value} (类型: {type(value)})，计为 0")


                if not em and error_file_path:
                    error_entry = {
                        "id": record_id,
                        "question": question,
                        "original_answer": original_answers,
                        "predicted_answer": prediction,
                        "hits_at_1": em,
                        "token_f1": f1_scores[-1],
                        "containment": containment_scores[-1],
                        "fix_count": data.get("fix_count", "N/A")
                    }

                    if alias_data and answers_to_match != original_answers:
                         error_entry["answers_used_for_match (incl. aliases)"] = answers_to_match
                    error_records.append(error_entry)

            except json.JSONDecodeError as e:
                logging.error(f"解析第 {i+1} 行 JSON 时出错: {e}")
            except Exception as e:
                logging.error(f"处理第 {i+1} 行时发生意外错误: {e}")

    if total_records == 0:
        logging.warning("文件中没有找到有效的记录进行评估。")
        return


    avg_em = sum(em_scores) / total_records
    avg_f1 = sum(f1_scores) / total_records
    avg_containment = sum(containment_scores) / total_records
    average_metrics: Dict[str, float] = {f"avg_{key}": metric_accumulators[key] / total_records for key in metric_keys}


    print("\n--- 评估结果报告 ---")
    print(f"处理总记录数: {total_records}")
    print(f"评估模式: {'使用了提供的别名文件' if alias_data else '未使用别名文件'}")
    print("-" * 20)
    print("性能指标:")
    print(f"  Hits@1 (normalized exact match): {avg_em:.4f} ({avg_em:.2%})")
    print(f"  Token-level F1:                  {avg_f1:.4f}")
    print(f"  Containment (secondary):         {avg_containment:.4f}")
    print("-" * 20)
    print("平均资源消耗:")
    print(f"  平均 Logic Init Time:        {average_metrics['avg_logic_init_time']:.4f} s")
    print(f"  平均 Self-Correction Time:     {average_metrics['avg_self_adaptive_time']:.4f} s")
    print(f"  平均 Final Reasoning Time:   {average_metrics['avg_final_reasoning_time']:.4f} s")
    print(f"  平均 Total Processing Time:  {average_metrics['avg_total_processing_time']:.4f} s")
    print("-" * 10)
    print(f"  平均 Logic Init Tokens:      {average_metrics['avg_logic_init_tokens']:.2f}")
    print(f"  平均 Self-Correction Tokens:   {average_metrics['avg_self_adaptive_tokens']:.2f}")
    print(f"  平均 Final Reasoning Tokens: {average_metrics['avg_final_reasoning_tokens']:.2f}")
    print(f"  平均 Total Tokens Consumed:  {average_metrics['avg_total_tokens']:.2f}")
    print("-" * 10)
    print(f"  平均 Fix Count:            {average_metrics['avg_fix_count']:.2f}")
    print("-" * 20)


    if error_file_path and error_records:
        try:
            error_dir = os.path.dirname(error_file_path)
            if error_dir and not os.path.exists(error_dir):
                os.makedirs(error_dir)
                logging.info(f"创建错误文件目录: {error_dir}")

            with open(error_file_path, 'w', encoding='utf-8') as errfile:
                for record in error_records:
                    errfile.write(json.dumps(record, ensure_ascii=False) + '\n')
            logging.info(f"已将 {len(error_records)} 条错误记录保存到: {error_file_path}")
        except Exception as e:
            logging.error(f"保存错误文件时出错: {e}")

    logging.info("评估完成。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate short-answer QA predictions using normalized EM, token F1, and containment.")
    parser.add_argument("--input_file", type=str, help="需要评估的 JSONL 结果文件路径。")


    parser.add_argument("--alias_file", type=str, default=None,
                        help="(可选) 别名文件的路径 (JSON 格式)。如果提供且有效，将用于 EM 评估。")
    parser.add_argument("--error_file", type=str, default=None,
                        help="(可选) 用于保存评估错误记录的 JSONL 文件路径。")

    args = parser.parse_args()




    alias_data = None
    if args.alias_file:
        alias_data = load_aliases(args.alias_file)

        if alias_data is None:
             logging.warning(f"无法从 {args.alias_file} 加载别名数据，评估将不使用别名。")


    evaluate_results(args.input_file, alias_data, args.error_file)
