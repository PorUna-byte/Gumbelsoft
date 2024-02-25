# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import json
import time
import tqdm
import numpy as np
import torch
import threading
from transformers import  AutoTokenizer
from wm.utils import load_results_labels
from wm import (WmDetector ,NgramWmDetector, GseqWmDetector, 
                       MarylandDetectorNg, MarylandDetectorGseq, OpenaiDetectorNg, OpenaiDetectorGseq, 
                       DipmarkDetectorNg, DipmarkDetectorGseq, GumbelSoftDetectorNg, GumbelSoftDetectorGseq,
                       ITSDetectorNg, ITSDetectorGseq)

def get_args_parser():
    parser = argparse.ArgumentParser('Args', add_help=False)

    # model parameters
    parser.add_argument('--tokenizer', type=str, default='llama-2-7b-chat')

    # watermark parameters
    parser.add_argument('--method', type=str, default='none',
                        help='watermark detection method')
    parser.add_argument('--method_suffix', type=str, default="")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seeding', type=str, default='hash', 
                        help='seeding method for rng key generation as introduced in https://github.com/jwkirchenbauer/lm-watermarking')
    parser.add_argument('--ngram', type=int, default=3, 
                        help='watermark context width for rng key generation')
    parser.add_argument('--gamma', type=float, default=0.1, 
                        help='gamma for maryland(dipmark): proportion of (non-)greenlist tokens')
    parser.add_argument('--hash_key', type=int, default=35317, 
                        help='hash key for rng key generation')
    parser.add_argument('--scoring_method', type=str, default='none', 
                        help='method for scoring. choose between: \
                        none (score every tokens), v1 (score token when wm context is unique), \
                        v2 (score token when {wm context + token} is unique')
    parser.add_argument('--ref_count', type=int, default=100,
                        help='For Global sequence watermarking methods, this controls how many other alternative\
                        watermark keys we will try')
    parser.add_argument('--wmkey_len', type=int, default=256,
                        help='For Global sequence watermarking methods, this controls how long our watermark keys are,\
                        typically the length of watermark key is twice as the length of our generated contents')
    parser.add_argument('--max_gen_len', type=int, default=128, 
                        help='maximum generation length')
    parser.add_argument('--natural_text_path', type=str, default='data/c4_ref.json', help='The path to natural text distribution')

    # unbiasedness
    parser.add_argument('--shift_max', type=int, default=0, 
                        help='unique identifier to achieve unbiasedness')
    
    # expe parameters
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--output_dir', type=str, default='experiment_results')
    parser.add_argument('--unwatermarked', type=bool, default=False)
    parser.add_argument('--table_name', type=str, default='table1')
    parser.add_argument('--repetition', type=int, default=10)
    parser.add_argument('--json_path', type=str, default='experiment_results')
    parser.add_argument('--result_key', type=str, default='result',
                        help='key to access result in json dict')
    parser.add_argument('--label_key', type=str, default='label')
    
    return parser

def main(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    model_dir = os.environ.get("model_dir")
    model_name = args.tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir+model_name)
    
    start=time.time()
    # build watermark detector
    match args.method:
        case "marylandNg":
            detector = MarylandDetectorNg(tokenizer, args.seed, args.shift_max, args.ngram, args.seeding, args.hash_key, args.scoring_method, gamma=args.gamma)
        case "marylandGseq":
            detector = MarylandDetectorGseq(tokenizer, args.seed, args.shift_max, args.wmkey_len, gamma=args.gamma)
        case "openaiNg":
            detector = OpenaiDetectorNg(tokenizer, args.seed, args.shift_max, args.ngram, args.seeding, args.hash_key, args.scoring_method)
        case "openaiGseq":
            detector = OpenaiDetectorGseq(tokenizer, args.seed, args.shift_max, args.wmkey_len,)
        case "dipmarkNg":
            detector = DipmarkDetectorNg(tokenizer, args.seed, args.shift_max, args.ngram, args.seeding, args.hash_key, args.scoring_method, gamma=args.gamma)
        case "dipmarkGseq":
            detector = DipmarkDetectorGseq(tokenizer, args.seed, args.shift_max, args.wmkey_len, gamma=args.gamma)
        case "gumbelsoftNg":
            detector = GumbelSoftDetectorNg(tokenizer, args.seed, args.shift_max, args.ngram, args.seeding, args.hash_key, args.scoring_method)
        case "gumbelsoftGseq":
            detector = GumbelSoftDetectorGseq(tokenizer, args.seed, args.shift_max, args.wmkey_len)    
        case "ITSNg":
            detector = ITSDetectorNg(tokenizer, args.seed, args.shift_max, args.ngram, args.seeding, args.hash_key, args.scoring_method, ref_count=args.ref_count, natural_text_path=args.natural_text_path, max_gen_len=args.max_gen_len, wmkey_len=args.wmkey_len)
        case "ITSGseq":
            detector = ITSDetectorGseq(tokenizer, args.seed, args.shift_max, args.wmkey_len, ref_count=args.ref_count, natural_text_path=args.natural_text_path, max_gen_len=args.max_gen_len)
        case _:
            raise NotImplementedError(f"Detector {args.method_detect} not implemented.")
    end=time.time()
    
    if args.unwatermarked:            
        detime_path = os.path.join(args.output_dir, "unwatermarked", args.table_name)
    else:
        detime_path = os.path.join(args.output_dir, args.method+args.method_suffix, args.table_name)
    os.makedirs(detime_path, exist_ok=True)      
    
    
    def rep_detect(rep):
        if args.unwatermarked:
            result_path = os.path.join(args.json_path, "unwatermarked", args.table_name, "rep_"+rep, f"results_unwatermarked.jsonl")
        else:
            result_path = os.path.join(args.json_path, args.method+args.method_suffix, args.table_name, "rep_"+rep, f"results_{args.method}{args.method_suffix}.jsonl")
            
        results, _ = load_results_labels(json_path=result_path, result_key=args.result_key, label_key=args.label_key)
        print(f"Loaded {len(results)} results.")
        # evaluate
        if args.unwatermarked:
            score_path = os.path.join(args.output_dir, "unwatermarked", args.table_name, "rep_"+rep)
        else: 
            score_path = os.path.join(args.output_dir, args.method+args.method_suffix, args.table_name, "rep_"+rep)
            
        os.makedirs(score_path, exist_ok=True)
        with open(os.path.join(score_path, f"scores_{args.method}{args.method_suffix}.jsonl"), 'w') as f:
            for i, result in tqdm.tqdm(enumerate(results), total=len(results)):
                log_stat = {'text_index':i}
                shift, zscore, pvalue, ntoks = detector.get_szp_by_t(result, toks=None)                
                log_stat['shift'] = shift
                log_stat['zscore'] = zscore
                log_stat['pvalue'] = pvalue
                log_stat['score_tokens'] = ntoks
                f.write(json.dumps(log_stat)+"\n")
                f.flush()
        for tok in [40, 60, 80, 100]:
            with open(os.path.join(score_path, f"scores_{args.method}{args.method_suffix}_tok{tok}.jsonl"), 'w') as f:
                for i, result in tqdm.tqdm(enumerate(results), total=len(results)):
                    log_stat = {'text_index':i}
                    start = time.time()
                    shift, zscore, pvalue, ntoks = detector.get_szp_by_t(result, toks=tok)                
                    log_stat['shift'] = shift
                    log_stat['zscore'] = zscore
                    log_stat['pvalue'] = pvalue
                    log_stat['score_tokens'] = ntoks
                    f.write(json.dumps(log_stat)+"\n")
                    f.flush()
        
          
    # List to keep track of the thread objects
    threads = []      
    for rep in range(1, args.repetition+1):
        rep=str(rep)
        thread = threading.Thread(target=rep_detect, args=(rep))
        threads.append(thread)

    # Start all the threads
    for thread in threads:
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    print(f"All detection threads have finished execution for method:{args.method}")              

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)