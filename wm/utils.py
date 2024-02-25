from typing import Dict, List
import json

def format_prompts(prompts: List[Dict], prompt_type: str) -> List[str]:
    if prompt_type=='completion':
        PROMPT_DICT = {
            "prompt_input": (
                "{input}"
            ),
            "prompt_no_input": (
                "{input}"
            ),
        }
    elif prompt_type=='chat':
        PROMPT_DICT = {
            "prompt_input": (
                "### Human: {instruction}\n\n### Input:\n{input}\n\n### Assistant:"
            ),
            "prompt_no_input": (
                "### Human: {instruction}\n\n### Assistant:"
            )
        }
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    prompts = [
        prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
        for example in prompts
    ]
    return prompts

def load_prompts_labels(json_path: str, prompt_type: str) -> List[str]:
    with open(json_path, "r") as f:
        prompts_labels = json.loads(f.read())
        
    prompts = format_prompts(prompts_labels, prompt_type)
    labels = [data['output'] for data in prompts_labels]
    return prompts, labels

def load_results_labels(json_path: str, result_key: str='result', label_key='label') -> List[str]:
    with open(json_path, "r") as f:
        if json_path.endswith('.json'):
            datas = json.loads(f.read())
        else:
            datas = [json.loads(line) for line in f.readlines()] # load jsonl
    results = [data[result_key] for data in datas]
    labels = [data[label_key] for data in datas]
    return results, labels


def load_scores(json_path: str, key: str='pvalue') -> List[float]:
    with open(json_path, "r") as f:
        if json_path.endswith('.json'):
            datas = json.loads(f.read())
        else:
            datas = [json.loads(line) for line in f.readlines()] # load jsonl
    scores = [data[key] for data in datas]
    return scores

def sample_texts(nsamples: int=None, data_path="data/c4_ref.json")->List[str]:
    with open(data_path, "r") as f:
        natural_texts = json.loads(f.read())
    return [text['input']+ " " +text['output'] for text in natural_texts][:nsamples]
