# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import time
import json

import numpy as np

import torch  
from transformers import AutoModelForCausalLM, AutoTokenizer

from wm import (Generator, NgramWmGenerator, GseqWmGenerator, 
                        MarylandGeneratorNg, MarylandGeneratorGseq, OpenaiGeneratorNg, OpenaiGeneratorGseq, 
                        DipmarkGeneratorNg, DipmarkGeneratorGseq, GumbelSoftGeneratorNg, GumbelSoftGeneratorGseq,
                        ITSGeneratorNg, ITSGeneratorGseq)

from wm.utils import load_prompts_labels

def get_args_parser():
    parser = argparse.ArgumentParser('Args', add_help=False)

    # model parameters
    parser.add_argument('--model_name', type=str)

    # prompts parameters
    parser.add_argument('--json_path', type=str, default="data/curated_data.json")
    parser.add_argument('--prompt_type', type=str, default="completion", 
                        help='type of prompt formatting. Choose between: completion/chat')
    # generation parameters
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--top_p', type=float, default=1)
    parser.add_argument('--max_gen_len', type=int, default=128)

    # watermark parameters
    parser.add_argument('--method', type=str, default='unwatermarked')
    parser.add_argument('--method_suffix', type=str, default="")
    parser.add_argument('--seeding', type=str, default='hash', 
                        help='seeding method for rng key generation as introduced in https://github.com/jwkirchenbauer/lm-watermarking')
    parser.add_argument('--ngram', type=int, default=3, 
                        help='watermark context width for rng key generation')
    parser.add_argument('--gamma', type=float, default=0.1, 
                        help='gamma for maryland/dipmark: proportion of (non)greenlist tokens')
    parser.add_argument('--alpha', type=float, default=0.45, 
                        help='alpha for Dipmark: probability proportion of tokens to be zeroed')
    parser.add_argument('--delta', type=float, default=2.0, 
                        help='delta for maryland: bias to add to greenlist tokens')
    parser.add_argument('--hash_key', type=int, default=35317, 
                        help='hash key for rng key generation')
    parser.add_argument('--wmkey_len', type=int, default=256,
                        help='For Global sequence watermarking methods, this controls how long our watermark keys are,\
                        typically the length of watermark key is twice as the length of our generated contents')
    parser.add_argument('--drop_prob', type=float, default=0,
                        help='The probability to choose randomly sample a token based on original probability distribution while doing exponential minimum sampling.')
    parser.add_argument('--tau', type=float, default=0)
    # unbiasedness
    parser.add_argument('--shift_max', type=int, default=0,
                        help='We use 0<=r<shift_max+1 and position t to decide the watermark key.\
                            the identifier r is used to achieve unbiasedness.')
    
    # expe parameters
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='experiment_results')
    parser.add_argument('--table_name', type=str, default='table1')
    parser.add_argument('--repetition', type=int, default=10)
    
    # attack
    parser.add_argument('--attack_name', type=str, default='none',
                        help='attack name to be applied to text before evaluation. Choose between: \
                        none (no attack), tok_substitution (randomly substitute tokens)')
    parser.add_argument('--attack_param', type=float, default=0,
                        help='attack parameter. For tok_substitution, it is the probability of substitution')
    
    # distributed parameters
    parser.add_argument('--ngpus', type=int, default=None)
    return parser

def main(args):

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    model_name = args.model_name
    model_dir = os.environ.get("model_dir")
    tokenizer = AutoTokenizer.from_pretrained(model_dir+model_name)
    args.ngpus = torch.cuda.device_count() if args.ngpus is None else args.ngpus
    print(f'device_count={torch.cuda.device_count()}')
    model = AutoModelForCausalLM.from_pretrained(
        model_dir+model_name,
        device_map="auto",
        # torch_dtype=torch.float16,
        # max_memory={i: '24000MB' for i in range(args.ngpus)},
        # offload_folder="offload",
    )

    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # build watermark generator
    match args.method:
        case "unwatermarked":
            generator = Generator(model, tokenizer, args.seed, args.shift_max, args.attack_name, args.attack_param)
        case "marylandNg":
            generator = MarylandGeneratorNg(model, tokenizer, args.seed, args.shift_max, args.attack_name, args.attack_param, args.ngram, args.seeding, args.hash_key, gamma=args.gamma, delta=args.delta)
        case "marylandGseq":
            generator = MarylandGeneratorGseq(model, tokenizer, args.seed, args.shift_max, args.attack_name, args.attack_param, args.wmkey_len, gamma=args.gamma, delta=args.delta)
        case "openaiNg":
            generator = OpenaiGeneratorNg(model, tokenizer, args.seed, args.shift_max, args.attack_name, args.attack_param, args.ngram, args.seeding, args.hash_key, drop_prob=args.drop_prob, tau=args.tau)
        case "openaiGseq":
            generator = OpenaiGeneratorGseq(model, tokenizer, args.seed, args.shift_max, args.attack_name, args.attack_param, args.wmkey_len, drop_prob=args.drop_prob)
        case "dipmarkNg":
            generator = DipmarkGeneratorNg(model, tokenizer, args.seed, args.shift_max, args.attack_name, args.attack_param, args.ngram, args.seeding, args.hash_key, alpha=args.alpha)
        case "dipmarkGseq":
            generator = DipmarkGeneratorGseq(model, tokenizer, args.seed, args.shift_max, args.attack_name, args.attack_param, args.wmkey_len, alpha=args.alpha)
        case "gumbelsoftNg":
            generator = GumbelSoftGeneratorNg(model, tokenizer, args.seed, args.shift_max, args.attack_name, args.attack_param, args.ngram, args.seeding, args.hash_key, drop_prob=args.drop_prob, tau=args.tau)
        case "gumbelsoftGseq":
            generator = GumbelSoftGeneratorGseq(model, tokenizer, args.seed, args.shift_max, args.attack_name, args.attack_param, args.wmkey_len)
        case "ITSNg":
            generator = ITSGeneratorNg(model, tokenizer, args.seed, args.shift_max, args.attack_name, args.attack_param, args.ngram, args.seeding, args.hash_key)
        case "ITSGseq":
            generator = ITSGeneratorGseq(model, tokenizer, args.seed, args.shift_max, args.attack_name, args.attack_param, args.wmkey_len)
        case _:
            raise NotImplementedError(f"method {args.method} is not implemented.")

    # load prompts and labels
    prompts, labels = load_prompts_labels(json_path=args.json_path, prompt_type=args.prompt_type)
    
    all_times = []
    # generate
    for rep in range(1, args.repetition+1):
        result_path = os.path.join(args.output_dir, args.method+args.method_suffix, args.table_name, "rep_"+str(rep))
        os.makedirs(result_path, exist_ok=True)
        with open(os.path.join(result_path, f"results_{args.method}{args.method_suffix}.jsonl"), "w") as f:
            for i in range(0, len(prompts), args.batch_size):
                # generate chunk
                chunk_size = min(args.batch_size, len(prompts) - i)
                time_pass, results = generator.generate(
                    prompts[i:i+chunk_size], 
                    max_gen_len=args.max_gen_len, 
                    temperature=args.temperature, 
                    top_p=args.top_p
                )
                print(f"Generated {i:5d} --- {i+chunk_size:5d}")
                all_times.append(time_pass)
                for prompt, result, label in zip(prompts[i:i+chunk_size], results, labels[i:i+chunk_size]):
                    f.write(json.dumps({
                        "prompt": prompt, 
                        "result": result,
                        "label": label}) + "\n")
                    f.flush()
        
    gen_path = os.path.join(args.output_dir, args.method+args.method_suffix, args.table_name, f"{args.method+args.method_suffix}_gentime.txt")                         
    with open(gen_path, "w") as f:
        f.write(f"Average generation time per prompt: {float(np.sum(all_times)) / (args.batch_size*len(all_times)) :.2f}\n")
        
if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
