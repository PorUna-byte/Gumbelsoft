import os
import argparse
import json

def get_args_parser():
    parser = argparse.ArgumentParser('Args', add_help=False)
    parser.add_argument('--experiments_path', type=str, default=os.environ.get("experiments_path"))
    parser.add_argument('--methods', type=str, default="marylandNg#openaiNg#dipmarkNg#gumbelsoftNg#ITSNg")
    parser.add_argument('--table_name', type=str, default="table1")
    parser.add_argument('--repetition', type=int, default=10)
    return parser

args = get_args_parser().parse_args()
methods = args.methods.split('#')

result_dict={}
for method in methods:
    result_dict[method]={'Avg_AUROC':{'tok_40':0.0,'tok_60':0.0,'tok_80':0.0,'tok_100':0.0}, 'Avg_FPR':{'tok_40':0.0,'tok_60':0.0,'tok_80':0.0,'tok_100':0.0}, 'Avg_FNR':{'tok_40':0.0,'tok_60':0.0,'tok_80':0.0,'tok_100':0.0}, 'Avg_ppl':0.0, 'Avg_sbert':0.0, 'Avg_rougeLf1':0.0}
result_dict['unwatermarked'] = {'Avg_ppl':0.0, 'Avg_sbert':0.0, 'Avg_rougeLf1':0.0}
   
for rep in range(1, args.repetition+1):
    rep_path = os.path.join(args.experiments_path, args.table_name)
    os.makedirs(rep_path, exist_ok=True)
    fw_rep = open(os.path.join(rep_path, f"rep{rep}.txt"), "w")
    rep=str(rep)
    for method in methods+["unwatermarked"]:
        fr_ppl = open(os.path.join(args.experiments_path, method, args.table_name, "rep_"+rep, "ppl.txt"))        
        fw_rep.write(f"{method}:\n")
        if method != "unwatermarked":
            for tok in [40, 60, 80, 100]:
                fw_rep.write(f"For tok_{tok}:\n")
                fr_acc = open(os.path.join(args.experiments_path, method, args.table_name, "rep_"+rep, f"acc_tok{tok}.txt"))
                for line in fr_acc.readlines():
                    fw_rep.write(line)
                    if line.split(':')[0]== "AUROC is":
                        result_dict[method]['Avg_AUROC'][f'tok_{tok}'] += float(line.split(':')[1])
                    elif line.split(':')[0]== "FPR is":
                        result_dict[method]['Avg_FPR'][f'tok_{tok}'] += float(line.split(':')[1])
                    elif line.split(':')[0]== "FNR is":
                        result_dict[method]['Avg_FNR'][f'tok_{tok}'] += float(line.split(':')[1])
                fw_rep.write("\n\n")

        for line in fr_ppl.readlines():
            fw_rep.write(line)
            if line.split(':')[0] == "Average ppl is":
                result_dict[method]['Avg_ppl'] += float(line.split(':')[1])
            elif line.split(':')[0] == "Average similarity_sbert is":
                result_dict[method]['Avg_sbert'] += float(line.split(':')[1])            
            elif line.split(':')[0] == 'Average rouge_f1 is':
                result_dict[method]['Avg_rougeLf1'] += float(line.split(':')[1])  
                 
        fw_rep.write("-"*50+"\n")
        
for method in methods:
    for tok in [40, 60, 80, 100]:
        result_dict[method]['Avg_AUROC'][f'tok_{tok}'] = round(result_dict[method]['Avg_AUROC'][f'tok_{tok}']/args.repetition, 3)
        result_dict[method]['Avg_FPR'][f'tok_{tok}'] = round(result_dict[method]['Avg_FPR'][f'tok_{tok}']/args.repetition, 3)
        result_dict[method]['Avg_FNR'][f'tok_{tok}'] = round(result_dict[method]['Avg_FNR'][f'tok_{tok}']/args.repetition, 3)
    result_dict[method]['Avg_ppl'] = round(result_dict[method]['Avg_ppl']/args.repetition, 3)
    result_dict[method]['Avg_sbert'] = round(result_dict[method]['Avg_sbert']/args.repetition, 3)
    result_dict[method]['Avg_rougeLf1'] = round(result_dict[method]['Avg_rougeLf1']/args.repetition, 3)
    
result_dict['unwatermarked']['Avg_ppl'] = round(result_dict['unwatermarked']['Avg_ppl']/args.repetition, 3)
result_dict['unwatermarked']['Avg_sbert'] = round(result_dict['unwatermarked']['Avg_sbert']/args.repetition, 3)
result_dict['unwatermarked']['Avg_rougeLf1'] = round(result_dict['unwatermarked']['Avg_rougeLf1']/args.repetition, 3)

fw = open(os.path.join(args.experiments_path, f"{args.table_name}.txt"), "w")
json.dump(result_dict, fw, indent=4)