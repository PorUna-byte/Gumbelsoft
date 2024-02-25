import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.util import ngrams
from collections import Counter
from nltk.tokenize import word_tokenize

import os
import json
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('Args', add_help=False)
    parser.add_argument('--experiments_path', type=str, default=os.environ.get("experiments_path"))
    parser.add_argument('--methods', type=str, default="openaiNg#openaiNg_s50#gumbelsoftNg#unwatermarked")
    parser.add_argument('--table_name', type=str, default="com_diverse")
    return parser

def calculate_self_bleu(completions):
    scores = []
    for i in range(len(completions)):
        hypothesis = completions[i]
        references = completions[:i] + completions[i+1:]
        # Tokenizing the sentences
        hypothesis_tokens = word_tokenize(hypothesis)
        reference_tokens = [word_tokenize(ref) for ref in references]
        # Calculating BLEU score
        score = sentence_bleu(reference_tokens, hypothesis_tokens, smoothing_function=SmoothingFunction().method1)
        scores.append(score)
    # Averaging the scores
    average_score = sum(scores) / len(scores)
    return average_score

def calculate_distinct_ngrams(completions, n):
    ngrams_all = []
    for completion in completions:
        tokens = nltk.word_tokenize(completion)
        ngrams_all.extend(list(ngrams(tokens, n)))
    ngram_counts = Counter(ngrams_all)
    return len(ngram_counts) / len(ngrams_all)

args = get_args_parser().parse_args()
methods = args.methods.split('#')
rep='1'
for method in methods:
    completions = []
    diversity_metrics = []
    fr_result_path = os.path.join(args.experiments_path, method, args.table_name, "rep_"+rep, f"results_{method}.jsonl")
    fw_result_path = os.path.join(args.experiments_path, method, args.table_name, "rep_"+rep, f"results_{method}_bleu.jsonl")
    fr_result = open(fr_result_path)
    fw_result = open(fw_result_path, "w")
    datas = [json.loads(line) for line in fr_result.readlines()] # load jsonl
    
    for i in range(len(datas)+1):
        if i%50 == 0 and i!=0:
            diversity_metrics.append({'prompt':datas[i-1]['prompt'], 'self_bleu':calculate_self_bleu(completions), 
            'Dist-1':calculate_distinct_ngrams(completions, 1), 'Dist-2':calculate_distinct_ngrams(completions, 2)})
            completions = []
            if i==len(datas):
                break 
        completions.append(datas[i]['result'])
        
    json.dump(diversity_metrics, fw_result, indent=4)
    
    fw_info = open(os.path.join(args.experiments_path, method, args.table_name, "rep_"+rep, "bleu.txt"), "w")
    sum_bleu = 0.0
    sum_dist1 = 0.0
    sum_dist2 = 0.0
    for metirc in diversity_metrics:
        sum_bleu += metirc['self_bleu']
        sum_dist1 += metirc['Dist-1']
        sum_dist2 += metirc['Dist-2']
    fw_info.write(f"Average bleu is:{sum_bleu/len(diversity_metrics):.3f}\n")
    fw_info.write(f"Average Dist-1 is:{sum_dist1/len(diversity_metrics):.3f}\n")
    fw_info.write(f"Average Dist-2 is:{sum_dist2/len(diversity_metrics):.3f}\n")


