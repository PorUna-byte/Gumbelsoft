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
    parser.add_argument('--repetition', type=int, default=10)
    
    return parser

def quartile(data_path):
    data = np.array(load_scores(data_path, key='zscore')) 
    # Calculate the quartiles
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    median = np.median(data)
    # Calculate the interquartile range (IQR)
    IQR = Q3 - Q1
    # Calculate the whiskers
    lower_whisker = data[data >= Q1 - 1.5 * IQR].min()
    upper_whisker = data[data <= Q3 + 1.5 * IQR].max()
    return round(lower_whisker,3), round(Q1,3), round(median,3), round(Q3,3), round(upper_whisker,3)

args = get_args_parser().parse_args()
methods = args.methods.split('#')
quartile_dict = {}
for rep in range(1, args.repetition+1):
    rep=str(rep)
    quartile_dict[rep]={}
    for method in methods:
        quartile_dict[rep][method]={}
        for tok in [40]:
            watermarked_path_unattacked = os.path.join(args.experiments_path, method, args.unattacked_table_name, "rep_"+rep, f"scores_{method}_tok{tok}.jsonl")
            watermarked_path_attacked = os.path.join(args.experiments_path, method, args.attacked_table_name, "rep_"+rep, f"scores_{method}_tok{tok}.jsonl")
            unwatermarked_path = os.path.join(args.experiments_path, "unwatermarked", args.unattacked_table_name, "rep_"+rep, f"scores_{method}_tok{tok}.jsonl")
            lower_whisker, Q1, median, Q3, upper_whisker=quartile(watermarked_path_unattacked)
            quartile_dict[rep][method]["watermarked_unattacked"]={"lower_whisker":lower_whisker, "Q1":Q1, "median":median, "Q3":Q3, "upper_whisker":upper_whisker}
            
            lower_whisker, Q1, median, Q3, upper_whisker=quartile(watermarked_path_attacked)
            quartile_dict[rep][method]["watermarked_attacked"]={"lower_whisker":lower_whisker, "Q1":Q1, "median":median, "Q3":Q3, "upper_whisker":upper_whisker}
            
            ower_whisker, Q1, median, Q3, upper_whisker=quartile(unwatermarked_path)
            quartile_dict[rep][method]["unwatermarked"]={"lower_whisker":lower_whisker, "Q1":Q1, "median":median, "Q3":Q3, "upper_whisker":upper_whisker}

for method in methods:
    quartile_dict[method]={}
    quartile_dict[method]["watermarked_unattacked"]={"lower_whisker":0, "Q1":0, "median":0, "Q3":0, "upper_whisker":0}
    quartile_dict[method]["watermarked_attacked"]={"lower_whisker":0, "Q1":0, "median":0, "Q3":0, "upper_whisker":0}
    quartile_dict[method]["unwatermarked"]={"lower_whisker":0, "Q1":0, "median":0, "Q3":0, "upper_whisker":0}
    for rep in range(1, args.repetition+1):
        for att_type in ["watermarked_unattacked", "watermarked_attacked", "unwatermarked"]:
            quartile_dict[method][att_type]["lower_whisker"]+=quartile_dict[str(rep)][method][att_type]["lower_whisker"]
            quartile_dict[method][att_type]["Q1"]+=quartile_dict[str(rep)][method][att_type]["Q1"]
            quartile_dict[method][att_type]["median"]+=quartile_dict[str(rep)][method][att_type]["median"]
            quartile_dict[method][att_type]["Q3"]+=quartile_dict[str(rep)][method][att_type]["Q3"]
            quartile_dict[method][att_type]["upper_whisker"]+=quartile_dict[str(rep)][method][att_type]["upper_whisker"]
    
    for att_type in ["watermarked_unattacked", "watermarked_attacked", "unwatermarked"]:
        quartile_dict[method][att_type]["lower_whisker"]=round(quartile_dict[method][att_type]["lower_whisker"]/args.repetition,3)
        quartile_dict[method][att_type]["Q1"]=round(quartile_dict[method][att_type]["Q1"]/args.repetition,3)
        quartile_dict[method][att_type]["median"]=round(quartile_dict[method][att_type]["median"]/args.repetition,3)
        quartile_dict[method][att_type]["Q3"]=round(quartile_dict[method][att_type]["Q3"]/args.repetition,3)
        quartile_dict[method][att_type]["upper_whisker"]=round(quartile_dict[method][att_type]["upper_whisker"]/args.repetition,3)
    
              
fw = open(os.path.join(args.experiments_path, f"robustness_{args.task}.txt"),"w")
json.dump(quartile_dict, fw, indent=4)
            
            
            
            

