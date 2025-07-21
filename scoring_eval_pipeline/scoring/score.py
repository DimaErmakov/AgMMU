import json
import os
import sys
from statistics import harmonic_mean
import argparse
from tqdm import tqdm
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils
import traceback
#TODO: change the paths to the supporting files, I'm lazy

with open('/u/dermakov/AgMMU/scoring_eval_pipeline/scoring/supporting_files/multi_statement.json') as file:
    multi = json.load(file)
# with open('/u/dermakov/AgMMU/scoring_eval_pipeline/scoring/supporting_files/few_word_examples.json') as file:
#     few_word_examples = json.load(file)
with open('/u/dermakov/AgMMU/scoring_eval_pipeline/scoring/supporting_files/examples_s1.json') as file:
    examples_s1 = json.load(file)
with open('/u/dermakov/AgMMU/scoring_eval_pipeline/scoring/supporting_files/examples_s2.json') as file:
    examples_s2 = json.load(file)
with open('/u/dermakov/AgMMU/scoring_eval_pipeline/scoring/supporting_files/few_word_examples.json') as file:
    few_word_examples = json.load(file)
# Scoring functions

def score_pipeline(data, output):
    ids = []
    if os.path.exists(output):
        with open(output) as file:
            ids_file = json.load(file)
        ids = [i['faq-id'] for i in ids_file]

    for q in tqdm(data, desc="Scoring questions"):
        if q['faq-id'] in ids:
            print(q['faq-id'])
            continue
        try:
            for llm in q['llm_answers']:
                question_block = q
                try:
                    if 'mcq' in llm:
                        if 'letter' in question_block:
                            q['llm_answers'][llm]['score'] = score_mcq({question_block['letter']: 1}, q['llm_answers'][llm]['answer'])
                        else:
                            # Don't ask me why but sometimes the letter is not in the question block
                            options = question_block['agmmu_question']['options']
                            answer = question_block['agmmu_question']['answer']
                            mapping = {}
                            for i, option in enumerate(options):
                                mapping[chr(65 + i)] = 1 if option == answer else 0
                            q['llm_answers'][llm]['score'] = score_mcq(mapping, q['llm_answers'][llm]['answer'])
                        # print(q['llm_answers'][llm]['score'])
                    elif q['qtype'] in ['management instructions', 'symptom/visual description']:
                        qset = 'management instructions' if q['qtype'] == 'management instructions' else (
                            "image description" if 'image description' in q['qa_information'] else 'symptom description'
                        )
                        if not isinstance(q['qa_information'][qset], list):
                            print("not a list", qset, q['qa_information'][qset])
                        res = score_multi_statement(q['qtype'], q['llm_answers'][llm]['answer'], q['qa_information'][qset])
                        q['llm_answers'][llm]['score'] = res
                        # print(q['llm_answers'][llm]['score'])

                    else:
                        q['llm_answers'][llm]['score'] = score_few_word(
                            question_block['question'],
                            question_block['answer'],
                            q['llm_answers'][llm]['answer'],
                            q['qtype'],
                            question_block['agmmu_question']['answer']
                        )
                        # print(q['llm_answers'][llm]['score'])

                except Exception as e:
                    print(f"error: {e}")
                    continue

            utils.add_item_to_json(output, q)
        except Exception as e:

            continue


def score_few_word(question, target, predicted_answer, qtype, agmmu_answer):
    # print(predicted_answer.strip().lower().replace(".", ""))
    # print(target.strip().lower().replace(".", ""))

    if predicted_answer.strip().lower().replace(".", "") == agmmu_answer.strip().lower().replace(".", ""):
        return {"accuracy": 1}
    elif predicted_answer.strip().lower().replace(".", "") + " plant" == agmmu_answer.strip().lower().replace(".", ""):
        return {"accuracy": 1}
    elif predicted_answer.strip().lower().replace(".", "") + " tree" == agmmu_answer.strip().lower().replace(".", ""):
        return {"accuracy": 1}
    elif predicted_answer.strip().lower().replace(".", "") == agmmu_answer.strip().lower().replace(".", "") + " plant":
        return {"accuracy": 1}
    elif predicted_answer.strip().lower().replace(".", "") + " tree" == agmmu_answer.strip().lower().replace(".", "") + " tree":
        return {"accuracy": 1}

    examples = ""
    system = "You are a helpful AI assistant."
    for i, example in enumerate(few_word_examples[qtype]):
        examples += f"EXAMPLE {i + 1}:\n\nQuestion:\n{example['question']}\nGold Target:\n{example['target']}\nPredicted Answer:\n{example['actual']}\nGrade:\n{example['grade']}\n  -{example['rational']}\n"

    prompt = f"""
     Your job is to grade student answers from the agriculture and biology domain. Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ['CORRECT', 'INCORRECT', 'NOT ATTEMPTED', 'PARTIALLY CORRECT'].
     First, I will give examples of each grade, and then you will grade a new example.
     {examples}

    Remember the following key points:
        - a statement should be AT LEAST partially correct if the predicted answer is a subcategory of the gold target or the gold target is a subcategory of the predicted answer
        - a statement is always partially correct if it has ANY overlap in content with the target

    Grade the predicted answer of this new question as one of:
    A: CORRECT
    B: INCORRECT
    C: NOT_ATTEMPTED
    D: PARTIALLY CORRECT

    Question: {question}
    Gold Target: {target}
    Predicted Answer: {predicted_answer}

    Just return the letters "A", "B", "C", or "D", with no text around it.
    """
    response = utils.exponential_backoff(utils.chat_gpt, system, prompt,None)
    filter_map = {"A": 1, "B": 0, "C": 0, "D": 0.5}
    return score_mcq(filter_map, response)

def create_multi_examples(qtype,question):
    if qtype == 'image description' or qtype == 'symptom description':
        qtype = "symptom/visual description"
    st = ""
    for i in multi[qtype]:
        st += f"Gold Target:\n{i['expected']}\nPredicted Answer:\n{i['actual']}\nScoring:\n{i['score']}\n"
    return st
def score_multi_statement(qtype, actual, expected):
    
    if qtype == 'management instructions':
        question = "What is the recommended management strategy for the issue seen in this image?"
    else:
        question ="What visual features can be seen in this image?" 
    examples = create_multi_examples(qtype,question)
    system = f"""
    Your job is to grade student answers from the agriculture and biology domain. Your job is to look at a question, a gold target, and a predicted answer, and then assign grades to each statement in the response of  ['correct','partially correct', 'incorrect', 'missing', 'irrelevant'].
        - Correct is assigned to statements from the predicted answer that fully semantically map to a statement in the gold target.
        - Partially correct is assigned to statements which partially semantically map to a statement in the gold target.
        - Incorrect is assigned to statements from the predicted answer that directly semantically contradict a statement in the gold target.
        - Missing is assigned to statements in the gold target which haven't been mapped within correct,partially correct, or incorrect. 
        - Irrelevant is assigned to statements in the predicted answer which neither directly contradict nor corrospond in any way to statements in the gold target.

    EACH STATEMENT IN THE GOLD TARGET AND PREDICTED ANSWER SHOULD BE ASSIGNED TO EXACTLY ONE OF THESE CATEGORIES.
    Here are examples of correctly graded statements:
    {examples}

    Remember the following key points:
        - a statement is always partially correct if it has ANY overlap in content with the target


    Question: {question}
    Gold Target: {expected}
    Predicted Answer: {actual}

    Follow the format of the examples exactly. Output only a json with no additional text.
    """

    prompt = f"Question: {question}\nActual Statement:\n{actual}\n True Statement(s):\n{expected}\nScoring:\n"

    response = utils.exponential_backoff(utils.chat_gpt, system, prompt,None)
    response = utils.clean_response(response)
    print(response)
    # I need to make sure that the response is a valid json so it does not break the code
    required_keys = {
        "correct": {},
        "incorrect": {},
        "partially correct": {},
        "missing": [],
        "irrelevant": [],
        "repeat": {}
    }
    for key, empty_val in required_keys.items():
        if key not in response:
            response[key] = empty_val

    return response



def score_mcq(target_map, predicted):
    cleaned = predicted.split(" ")[0].strip().lower().replace(".", "")
    for target in target_map:
        if cleaned == target.strip(" ")[0].lower().replace(".", "").strip():
            return {"accuracy": target_map[target]}
    return {"accuracy": 0}
        
    return data
def ensure_list(x):
    if isinstance(x, dict):
        return list(x.values())
    elif isinstance(x, list):
        return x
    else:
        return [x]

def get_stats(data):
    scores = {}
    for i in data:
        for llm in i['llm_answers']:
            scores.setdefault(llm, {}).setdefault(i['qtype'], {"correct": 0, "total": 0, "partial": 0, "num_questions": 0})
            metrics = scores[llm][i['qtype']]

            if 'mcq' in llm or i['qtype'] not in ['management instructions', 'symptom/visual description']:
                acc = i['llm_answers'][llm]['score'].get('accuracy', 0)
                if acc == 1:
                    metrics['correct'] += 1
                elif acc == 0.5:
                    metrics['partial'] += 1
                metrics['total'] += 1
                metrics['num_questions'] += 1
            else:
                temp = i['llm_answers'][llm]['score']
                target_num = (
                    ensure_list(temp.get('correct', {}))
                    + ensure_list(temp.get('partially correct', {}))
                    + ensure_list(temp.get('incorrect', {}))
                    + ensure_list(temp.get('missing', []))
                )
                num_statements = len(target_num)
                metrics['correct'] += len(temp.get('correct', {})) / num_statements if num_statements else 0
                metrics['partial'] += len(temp.get('partially correct', {})) / num_statements if num_statements else 0
                metrics['total'] += 1
                metrics['num_questions'] += 1
    return scores


def calculate_harmonic_means(data):
    result = {}
    for model, categories in data.items():
        result[model] = {}
        for category, metrics in categories.items():
            correct = metrics['correct']
            total = metrics['num_questions']
            partial = metrics.get('partial', 0)
            metric1 = correct / total if total > 0 else 0
            metric2 = correct / (total - partial) if (total - partial) > 0 else 0
            result[model][category] = harmonic_mean([metric1, metric2]) if metric1 > 0 and metric2 > 0 else 0
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Path to input JSON")
    parser.add_argument("--output", "-o", required=True, help="Path to output JSON")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    score_pipeline(data, args.output)
    with open(args.output) as file:
        archived_data = json.load(file)

    stats = get_stats(archived_data)
    harmonic_results = calculate_harmonic_means(stats)
    print("Harmonic Means:", harmonic_results)
    
    # Use this code if you want to calculate the harmonic means for a single file without scoring the questions
    # testings = ["/u/dermakov/rag_output_file_7_10.json","/u/dermakov/output_file_originL.json"]
    # for testing in testings:
    #     print("Testing file:", testing)
    #     with open(testing) as file:
    #         archived_data = json.load(file)

    #     stats = get_stats(archived_data)
    #     harmonic_results = calculate_harmonic_means(stats)
    #     print("Harmonic Means:", harmonic_results)