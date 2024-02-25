import torch
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import shutil
import threading

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args_parser():
    parser = argparse.ArgumentParser('Args', add_help=False)
    parser.add_argument('--ppl_model', type=str, default="llama-2-13b-chat")
    parser.add_argument('--experiments_path', type=str, default=os.environ.get("experiments_path"))
    parser.add_argument('--methods', type=str, default="marylandNg#openaiNg#dipmarkNg#gumbelsoftNg#ITSNg")
    parser.add_argument('--table_name', type=str, default="table1")
    parser.add_argument('--repetition', type=int, default=5)
    return parser

def calculate_ppl(cur_prompt, cur_gen, model, tokenizer):
    tokd_all = tokenizer.encode(cur_prompt + cur_gen, return_tensors='pt').to(model.device)
    tokd_gen = tokenizer.encode(cur_gen, return_tensors='pt').to(model.device)
    tokd_labels = tokd_all.clone().detach()
    tokd_labels[:, :tokd_labels.shape[1]-tokd_gen.shape[1]+1] = -100
    with torch.no_grad():
        outputs = model(tokd_all, labels=tokd_labels)
        loss = outputs.loss
        ppl = torch.exp(loss)
    return ppl.item()

model_dir = os.environ.get("model_dir")
args = get_args_parser().parse_args()
methods = args.methods.split('#')

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, args.ppl_model))
model = AutoModelForCausalLM.from_pretrained(os.path.join(model_dir, args.ppl_model), device_map="auto")
def evalppl_rep_method(rep, method):
    fr_result_path = os.path.join(args.experiments_path, method, args.table_name, "rep_"+rep, f"results_{method}.jsonl")
    fw_result_path = os.path.join(args.experiments_path, method, args.table_name, "rep_"+rep, f"results_{method}_ppl.jsonl")
    fr_result = open(fr_result_path)
    fw_result = open(fw_result_path, "w")
    datas = [json.loads(line) for line in fr_result.readlines()] # load jsonl

    all_ppl = []
    for i, data in enumerate(datas):   
        datas[i]["ppl"]=calculate_ppl(data['prompt'], data['result'], model, tokenizer)
        all_ppl.append(datas[i]["ppl"])
        fw_result.write(json.dumps(datas[i])+"\n")
        if i%200==0:
            print(f"rep:{rep}, method:{method}, {i+1}/{len(datas)}")
            fw_result.flush()
    fw_info = open(os.path.join(args.experiments_path, method, args.table_name, "rep_"+rep, "ppl.txt"), "w")
    fw_info.write(f"Average ppl is:{sum(all_ppl)/len(all_ppl):.3f}\n")
    shutil.move(fw_result_path, fr_result_path)

# for rep in range(1, args.repetition+1):
#     rep=str(rep)
#     for method in methods:
#         evalppl_rep_method(rep, method)
        

# List to keep track of the thread objects
threads = []      
for rep in range(1, args.repetition+1):
    rep=str(rep)
    for method in methods:
        thread = threading.Thread(target=evalppl_rep_method, args=(rep, method))
        threads.append(thread)

# Start all the threads
for thread in threads:
    thread.start()

# Wait for all threads to finish
for thread in threads:
    thread.join()

print("All threads have finished execution.")   