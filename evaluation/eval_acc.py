import os
import numpy as np
from evaluation.auroc import auroc_fpr_fnr
from wm.utils import load_scores
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('Args', add_help=False)
    parser.add_argument('--experiments_path', type=str, default=os.environ.get("experiments_path"))
    parser.add_argument('--methods', type=str, default="marylandNg#openaiNg#dipmarkNg#gumbelsoftNg#ITSNg")
    parser.add_argument('--table_name', type=str, default="table1")
    parser.add_argument('--repetition', type=int, default=10)
    return parser

args = get_args_parser().parse_args()
methods = args.methods.split('#')
def eval_acc(watermarked_path, unwatermarked_path, method, tok):
    watermarked_zscore = load_scores(watermarked_path, key='zscore')
    unwatermarked_zscore = load_scores(unwatermarked_path, key='zscore')
    #make sure watermarked is positive, that is the score for watermarked is higher
    scores = np.array(watermarked_zscore+unwatermarked_zscore)
    labels = np.zeros(len(scores))
    labels[:len(watermarked_zscore)] = 1
    auroc, fpr, fnr = auroc_fpr_fnr(labels, scores)
    fw = open(os.path.join(args.experiments_path, method, args.table_name, "rep_"+rep, f"acc_tok{tok}.txt"), "w")
    fw.write(f"AUROC is:{auroc:.3f}\nFPR is:{fpr:.3f}\nFNR is:{fnr:.3f}\n")

for rep in range(1, args.repetition+1):
    rep=str(rep)
    for method in methods:
        for tok in [40, 60, 80, 100]:
            watermarked_path = os.path.join(args.experiments_path, method, args.table_name, "rep_"+rep, f"scores_{method}_tok{tok}.jsonl")
            unwatermarked_path = os.path.join(args.experiments_path, "unwatermarked", args.table_name, "rep_"+rep, f"scores_{method}_tok{tok}.jsonl")
            eval_acc(watermarked_path, unwatermarked_path, method, tok)
