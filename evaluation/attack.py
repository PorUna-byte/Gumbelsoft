import os
import argparse
import json
import torch
import random
from transformers import T5Tokenizer, T5ForConditionalGeneration
import threading

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def get_args_parser():
    parser = argparse.ArgumentParser('Args', add_help=False)
    parser.add_argument('--experiments_path', type=str, default=os.environ.get("experiments_path"))
    parser.add_argument('--methods', type=str, default="marylandNg#openaiNg#dipmarkNg#gumbelsoftNg#ITSNg")
    parser.add_argument('--read_table_name', type=str, default="table1")
    parser.add_argument('--write_table_name', type=str, default="table2")
    parser.add_argument('--repetition', type=int, default=5)
    parser.add_argument('--attack_param', type=float, default=0.2)
    parser.add_argument('--attack_unwatermarked', type=bool, default=False)
    return parser

def attack(attack_model, attack_tokenizer, words):
    mid = 5
    words[mid] = '<extra_id_1>'
    masked_text = " ".join(words)
    input_ids = attack_tokenizer.encode(masked_text, return_tensors='pt').to(device)
    output = attack_model.generate(input_ids, max_length=5, num_beams=50, num_return_sequences=1)[0]
    predict_word = attack_tokenizer.decode(output, skip_special_tokens=True).split(" ")[0]
    return predict_word

model_dir = os.environ.get("model_dir")
args = get_args_parser().parse_args()
methods = args.methods.split('#')
if args.attack_unwatermarked is True:
    methods += ["unwatermarked"]
    
def attack_rep_method(rep, method):    
    attack_tokenizer = T5Tokenizer.from_pretrained(os.path.join(model_dir ,'t5-large'))
    attack_model = T5ForConditionalGeneration.from_pretrained(os.path.join(model_dir ,'t5-large')).to(device)
    print(f"repetition-{rep}, method-{method}")
    fr_result_path = os.path.join(args.experiments_path, method, args.read_table_name, "rep_"+rep, f"results_{method}.jsonl")
    os.makedirs(os.path.join(args.experiments_path, method, args.write_table_name, "rep_"+rep), exist_ok=True)
    fw_result_path = os.path.join(args.experiments_path, method, args.write_table_name, "rep_"+rep, f"results_{method}.jsonl")
    fr_result = open(fr_result_path)
    fw_result = open(fw_result_path, "w")
    unattacked_datas = [json.loads(line) for line in fr_result.readlines()] # load jsonl
    for i,data in enumerate(unattacked_datas):  
        result = data["result"]
        words = result.split(" ")
        if int((len(words)-10)*args.attack_param)<=0:
            fw_result.write(json.dumps(data)+"\n")
            fw_result.flush()
            continue

        # attack results, substitute self.attack_param percent words using its context and t5-large 
        attack_pos = random.sample(range(5,len(words)-5), int((len(words)-10)*args.attack_param))
        for pos in attack_pos:
            words[pos] = attack(attack_model, attack_tokenizer, words[pos-5:pos+6])
        data["result"]=" ".join(words)
        fw_result.write(json.dumps(data)+"\n")
        fw_result.flush()
        if i % 200 == 0:
            print(f"rep:{rep}, method:{method}, {i}/{len(unattacked_datas)}")
        
# List to keep track of the thread objects
threads = []      
for rep in range(1, args.repetition+1):
    rep=str(rep)
    for method in methods:
        thread = threading.Thread(target=attack_rep_method, args=(rep, method))
        threads.append(thread)

# Start all the threads
for thread in threads:
    thread.start()

# Wait for all threads to finish
for thread in threads:
    thread.join()

print("All threads have finished execution.")         