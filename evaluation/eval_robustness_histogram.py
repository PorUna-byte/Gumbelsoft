import os
import numpy as np
import argparse
from wm.utils import load_scores
import json
def get_args_parser():
    parser = argparse.ArgumentParser('Args', add_help=False)
    parser.add_argument('--experiments_path', type=str, default=os.environ.get("experiments_path"))
    parser.add_argument('--methods', type=str, default="marylandNg#openaiNg#dipmarkNg#gumbelsoftNg#ITSNg")
    parser.add_argument('--unattacked_table_name', type=str, default="table1")
    parser.add_argument('--attacked_table_name', type=str, default="table2")
    parser.add_argument('--task', type=str, default="Completion")
    return parser

args = get_args_parser().parse_args()
methods = args.methods.split('#')
zscore_dict = {}
for method in methods:
    zscore_dict[method]={}
    watermarked_path_unattacked = os.path.join(args.experiments_path, method, args.unattacked_table_name, "rep_1", f"scores_{method}_tok40.jsonl")
    watermarked_path_attacked = os.path.join(args.experiments_path, method, args.attacked_table_name, "rep_1", f"scores_{method}_tok40.jsonl")
    unwatermarked_path = os.path.join(args.experiments_path, "unwatermarked", args.unattacked_table_name, "rep_1", f"scores_{method}_tok40.jsonl")
    zscore_dict[method]["watermarked_unattacked"]=load_scores(watermarked_path_unattacked, key='zscore')
    zscore_dict[method]["watermarked_attacked"]=load_scores(watermarked_path_attacked, key='zscore')
    zscore_dict[method]["unwatermarked"]=load_scores(unwatermarked_path, key='zscore')
              
fw = open(os.path.join(args.experiments_path, f"robustness_{args.task}.json"),"w")
json.dump(zscore_dict, fw, indent=4)