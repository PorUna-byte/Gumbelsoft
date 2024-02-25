import os
import argparse
import json
def get_args_parser():
    parser = argparse.ArgumentParser('Args', add_help=False)
    parser.add_argument('--experiments_path', type=str, default=os.environ.get("experiments_path"))
    parser.add_argument('--methods', type=str, default="marylandNg#openaiNg#dipmarkNg#gumbelsoftNg#ITSNg")
    return parser

args = get_args_parser().parse_args()
methods = args.methods.split('#')

exp_path = args.experiments_path
for trail in ["com_diverse", "chat_diverse"]:
    fw=open(os.path.join(exp_path, trail+".txt"),"w")
    for method in methods:
        bleu_path = os.path.join(exp_path, method, trail, "rep_1", "bleu.txt")
        fr = open(bleu_path)
        fw.write(method+":\n")
        fw.write(fr.read())
        fw.write("\n")
        fw.write("#"*80)
        fw.write("\n")
    fw.write("\n\n")