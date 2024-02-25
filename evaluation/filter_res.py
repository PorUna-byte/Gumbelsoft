import json
import os

length=128
import argparse
import shutil

def get_args_parser():
    parser = argparse.ArgumentParser('Args', add_help=False)
    parser.add_argument('--experiments_path', type=str, default=os.environ.get("experiments_path"))
    parser.add_argument('--length', type=int, default=128)
    parser.add_argument('--stage', type=int, default=1)
    parser.add_argument('--ngram', type=int, default=3)
    parser.add_argument('--nsamples', type=int, default=1000)
    parser.add_argument('--methods', type=str, default="marylandNg#openaiNg#dipmarkNg#gumbelsoftNg#ITSNg")
    parser.add_argument('--table_name', type=str, default="table1")
    parser.add_argument('--repetition', type=int, default=10)
    return parser

args = get_args_parser().parse_args()
methods_ori = args.methods.split('#')

if args.stage==1:
    # make sure the results have enough tokens for ngram calculation.
    methods = methods_ori + ['unwatermarked']
    for rep in range(1, args.repetition+1):
        rep=str(rep)
        for method in methods:
            fr_result_path = os.path.join(args.experiments_path, method, args.table_name, "rep_"+rep, f"results_{method}.jsonl")
            fw_result_path = os.path.join(args.experiments_path, method, args.table_name, "rep_"+rep, f"results_{method}_fil.jsonl")
            fr_result = open(fr_result_path)
            fw_result = open(fw_result_path, "w")

            results = [json.loads(line) for line in fr_result.readlines()]
            filtered_results = []
            for result in results:
                if len(result["result"].split(" "))>2*args.ngram:
                    filtered_results.append(result)
                        
            for i, result in enumerate(filtered_results):
                if i==0:
                    fw_result.write(json.dumps(result))
                else:
                    fw_result.write("\n"+json.dumps(result))
                fw_result.flush()  

            os.remove(fr_result_path)
            shutil.move(fw_result_path, fr_result_path)

elif args.stage==2:
    for rep in range(1, args.repetition+1):
        rep=str(rep)
        # make sure the results have almost the same number of tokens
        methods = methods_ori
        # deal with watermarked score
        for method in methods:
            fr_score_path  = os.path.join(args.experiments_path, method, args.table_name, "rep_"+rep, f"scores_{method}.jsonl")
            fw_score_path  = os.path.join(args.experiments_path, method, args.table_name, "rep_"+rep, f"scores_{method}_final.jsonl")
            fr_result_path = os.path.join(args.experiments_path, method, args.table_name, "rep_"+rep, f"results_{method}.jsonl")
            fw_result_path = os.path.join(args.experiments_path, method, args.table_name, "rep_"+rep, f"results_{method}_final.jsonl")
            fr_score = open(fr_score_path)
            fw_score = open(fw_score_path, "w")
            fr_result = open(fr_result_path)
            fw_result = open(fw_result_path, "w")

            scores = [json.loads(line) for line in fr_score.readlines()] # load jsonl
            results = [json.loads(line) for line in fr_result.readlines()]
            filtered_scores = []
            for score in scores:
                if score["score_tokens"]>=args.length-5 and score["score_tokens"]<=args.length+5:
                    filtered_scores.append(score)
                    
            selected_indices=[]
            
            if filtered_scores[args.nsamples-1]["text_index"]>=len(results):
                os.remove(fw_score_path)
                os.remove(fw_result_path)
                continue
            
            for i, score in enumerate(filtered_scores[:args.nsamples]):
                selected_indices.append(score["text_index"])
                if i==0:
                    fw_score.write(json.dumps(score))
                    fw_result.write(json.dumps(results[score["text_index"]]))
                else:
                    fw_score.write("\n"+json.dumps(score))
                    fw_result.write("\n"+json.dumps(results[score["text_index"]]))
                fw_score.flush()
                fw_result.flush() 

            os.remove(fr_score_path)
            shutil.move(fw_score_path, fr_score_path)

            os.remove(fr_result_path)       
            shutil.move(fw_result_path, fr_result_path)

            for tok in [40, 60, 80, 100]:
                fr_score_path  = os.path.join(args.experiments_path, method, args.table_name, "rep_"+rep, f"scores_{method}_tok{tok}.jsonl")
                fw_score_path  = os.path.join(args.experiments_path, method, args.table_name, "rep_"+rep, f"scores_{method}_tok{tok}_final.jsonl")
                fr_score = open(fr_score_path)
                fw_score = open(fw_score_path, "w")

                scores = [json.loads(line) for line in fr_score.readlines()] # load jsonl
                if selected_indices[-1]>=len(scores):
                    os.remove(fw_score_path)
                    continue
                for i, index in enumerate(selected_indices):
                    if i==0:
                        fw_score.write(json.dumps(scores[index]))
                    else:
                        fw_score.write("\n"+json.dumps(scores[index]))
                    fw_score.flush()

                os.remove(fr_score_path)
                shutil.move(fw_score_path, fr_score_path)

        #deal with unwatermarked score
        for method in methods:
            fr_score_path  = os.path.join(args.experiments_path, "unwatermarked", args.table_name, "rep_"+rep, f"scores_{method}.jsonl")
            fw_score_path  = os.path.join(args.experiments_path, "unwatermarked", args.table_name, "rep_"+rep, f"scores_{method}_final.jsonl")
            fr_result_path = os.path.join(args.experiments_path, "unwatermarked", args.table_name, "rep_"+rep, f"results_unwatermarked.jsonl")
            fw_result_path = os.path.join(args.experiments_path, "unwatermarked", args.table_name, "rep_"+rep, f"results_unwatermarked_final.jsonl")
            fr_score = open(fr_score_path)
            fw_score = open(fw_score_path, "w")
            fr_result = open(fr_result_path)
            fw_result = open(fw_result_path, "w")
            
            scores = [json.loads(line) for line in fr_score.readlines()] # load jsonl
            results = [json.loads(line) for line in fr_result.readlines()]
            filtered_scores = []
            for score in scores:
                if score["score_tokens"]>=args.length-5 and score["score_tokens"]<=args.length+5:
                    filtered_scores.append(score)
                    
            if filtered_scores[args.nsamples-1]["text_index"]>=len(results):
                os.remove(fw_score_path)
                os.remove(fw_result_path)
                continue
            
            selected_indices=[]
            for i, score in enumerate(filtered_scores[:args.nsamples]):
                selected_indices.append(score["text_index"])
                if i==0:
                    fw_score.write(json.dumps(score))
                    fw_result.write(json.dumps(results[score["text_index"]]))
                else:
                    fw_score.write("\n"+json.dumps(score))
                    fw_result.write("\n"+json.dumps(results[score["text_index"]]))

                fw_score.flush()
                fw_result.flush() 
                
            os.remove(fr_score_path)
            shutil.move(fw_score_path, fr_score_path)

            for tok in [40, 60, 80, 100]:
                fr_score_path  = os.path.join(args.experiments_path, "unwatermarked", args.table_name, "rep_"+rep, f"scores_{method}_tok{tok}.jsonl")
                fw_score_path  = os.path.join(args.experiments_path, "unwatermarked", args.table_name, "rep_"+rep, f"scores_{method}_tok{tok}_final.jsonl")
                fr_score = open(fr_score_path)
                fw_score = open(fw_score_path, "w")
                scores = [json.loads(line) for line in fr_score.readlines()] # load jsonl
                
                if selected_indices[-1]>=len(scores):
                    os.remove(fw_score_path)
                    continue
                for i, index in enumerate(selected_indices):
                    if i==0:
                        fw_score.write(json.dumps(scores[index]))
                    else:
                        fw_score.write("\n"+json.dumps(scores[index]))

                    fw_score.flush()
                     
                os.remove(fr_score_path)
                shutil.move(fw_score_path, fr_score_path)
                
        # os.remove(fr_result_path)
        # shutil.move(fw_result_path, fr_result_path)