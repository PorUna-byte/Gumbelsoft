# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch
import numpy as np
import random
import time
import os
import time
from transformers import LlamaForCausalLM, LlamaTokenizer, T5Tokenizer, T5ForConditionalGeneration

model_dir = os.environ.get("model_dir")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Generator():
    """
    This is a general purpose generator, all other generator is a subclass of this generator
    """
    def __init__(self,
            model: LlamaForCausalLM,
            tokenizer: LlamaTokenizer,
            seed: int = 42,
            shift_max: int = 0,
            attack_name: str = None,
            attack_param: float = 0,
        ):
        self.model = model
        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size
        self.seed = seed
        self.shift_max = shift_max
        self.attack_name = attack_name
        self.attack_param = attack_param
        if attack_name == "tok_substitution":
            self.attack_tokenizer = T5Tokenizer.from_pretrained(os.path.join(model_dir ,'t5-large'))
            self.attack_model = T5ForConditionalGeneration.from_pretrained(os.path.join(model_dir ,'t5-large')).to(device)
        self.max_seq_len = model.config.max_position_embeddings
        self.pad_id = model.config.pad_token_id if model.config.pad_token_id is not None else 0
        self.eos_id = model.config.eos_token_id
        self.ngram = 0
        #for generator we use it to sample a torch.tensor
        self.rng = torch.Generator(device='cpu')

    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 1,
        top_p: float = 1,
    ) -> List[str]:
        """
        Generate text from prompts. 
        For each call to generate, we deem it as a response, and we assign an(almost) unique identifier for each response.
        Adapted from https://github.com/facebookresearch/llama/
        """
        bsz = len(prompts)
        prompt_tokens = [self.tokenizer.encode(x, add_special_tokens=False) for x in prompts]
        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])
        total_len = min(self.max_seq_len, max_gen_len + max_prompt_size)
        tokens = torch.full((bsz, total_len), self.pad_id).to(device).long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        
        time1 = time.time()
        unique_identifier = self.gen_unique_id(bsz)
        for cur_pos in range(start_pos, total_len):
            # Use LLM to calculate logits vector l
            outputs = self.model.forward(
                tokens[:, prev_pos:cur_pos], use_cache=True, past_key_values=outputs.past_key_values if prev_pos > 0 else None
            )
            ngram_tokens = tokens[:, cur_pos-self.ngram:cur_pos]
            xi = self.F_key(unique_identifier, ngram_tokens, cur_pos-start_pos)
            next_toks = self.Gamma(xi, outputs.logits[:, -1, :], temperature, top_p)
            tokens[:, cur_pos] = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_toks)
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[len(prompt_tokens[i]): len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            if t.count(self.eos_id) != 0:
                t = t[: t.index(self.eos_id)]
                    
            decoded.append(self.tokenizer.decode(t))
        time2 = time.time()
        if self.attack_name == 'tok_substitution':
            for i, text in enumerate(decoded):
                words = text.split(" ")
                if int((len(words)-10)*self.attack_param)<=0:
                    continue
                # attack results, substitute self.attack_param percent words using its context and t5-large 
                attack_pos = random.sample(range(5,len(words)-5), int((len(words)-10)*self.attack_param))
                for pos in attack_pos:
                    words[pos] = self.attack(words[pos-5:pos+6])
                    
                decoded[i] = " ".join(words)
                
        return time2-time1, decoded

    def attack(self, words):
        mid = 5
        words[mid] = '<extra_id_1>'
        masked_text = " ".join(words)
        input_ids = self.attack_tokenizer.encode(masked_text, return_tensors='pt').to(device)
        output = self.attack_model.generate(input_ids, max_length=5, num_beams=50, num_return_sequences=1)[0]
        predict_word = self.attack_tokenizer.decode(output, skip_special_tokens=True).split(" ")[0]
        return predict_word
        
    def gen_unique_id(self, bsz):
        """ Generate a unique identifier for each response """
        return np.random.randint(self.shift_max+1, size=bsz)
    
    def F_key(self, r, ngram, t):
        """ calculate the watermark key xi at position t, we use identifier r and position t/ngram
        to calculate a watermark key."""
        pass
    
    def top_p(self, logits, temperature, top_p):
        """ An utility function for top_p sampling """
        probs = torch.softmax(logits / temperature, dim=-1)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort >= top_p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token_probs = torch.zeros(logits.shape).to(device).scatter_(-1, probs_idx, probs_sort) # probability of next token, ordered by vocab 
        return next_token_probs
    
    def Gamma(
        self,
        xi, # the watermark key for current position
        logits: torch.FloatTensor, # (bsz, vocab_size): logits for last token
        temperature: float = 1, # temperature for sampling
        top_p: float = 1, # top p for sampling
    ) -> torch.LongTensor:
        """ This is the decoder: Take a watermark key xi and a logits vector l to decide the next token.
        Vanilla sampling with temperature and top p."""
        if temperature > 0:
            next_token_probs = self.top_p(logits, temperature, top_p)  
            next_token = torch.multinomial(next_token_probs, num_samples=1) # one hot of next token, ordered by original probs
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)
        return next_token

class NgramWmGenerator(Generator):
    """
    This kind of generator use previous Ngram and a hash function to calculate the watermark key.
    """
    def __init__(self, 
            model: LlamaForCausalLM, 
            tokenizer: LlamaTokenizer,
            seed: int = 42,
            shift_max : int = 0, 
            attack_name: str = None,
            attack_param: float = 0,
            ngram: int = 1,
            seeding: str = 'hash',
            hash_key: int = 35317,
        ):
        # model config
        super().__init__(model, tokenizer, seed, shift_max, attack_name, attack_param)
        # watermark config
        self.ngram = ngram
        self.seeding = seeding 
        self.hash_key = hash_key
        self.hashtable = torch.randperm(1000003)

    def hashint(self, integer_tensor: torch.LongTensor) -> torch.LongTensor:
        """Adapted from https://github.com/jwkirchenbauer/lm-watermarking"""
        return self.hashtable[integer_tensor.cpu() % len(self.hashtable)] 
    
    def get_seed_rng(
        self, 
        input_ids: torch.LongTensor
    ) -> int:
        """
        Seed RNG with hash of input_ids.
        Adapted from https://github.com/jwkirchenbauer/lm-watermarking
        """
        if self.seeding == 'hash':
            seed = self.seed
            for i in input_ids:
                seed = (seed * self.hash_key + i.item()) % (2 ** 64 - 1)
        elif self.seeding == 'additive':
            seed = self.hash_key * torch.sum(input_ids).item()
            seed = self.hashint(seed)
        elif self.seeding == 'skip':
            seed = self.hash_key * input_ids[0].item()
            seed = self.hashint(seed)
        elif self.seeding == 'min':
            seed = self.hashint(self.hash_key * input_ids)
            seed = torch.min(seed).item()
        return seed
    
class GseqWmGenerator(Generator):
    """
    This kind of generator use position t and a global watermark key list to calculate the watermark key.
    """
    def __init__(self, 
            model: LlamaForCausalLM, 
            tokenizer: LlamaTokenizer,
            seed: int = 42,    
            shift_max : int=0,
            attack_name: str = None,
            attack_param: float = 0,
            wmkey_len: int=256, 
        ):
        # model config
        super().__init__(model, tokenizer, seed, shift_max, attack_name, attack_param)
        # watermark config      
        self.wmkey_len = wmkey_len
        self.xis = []
    
    def F_key(self, r, ngram, t):
        """ calculate the watermark key xi at position t """
        xis = [self.xis[i] for i in r]
        return xis 

class MarylandGeneratorNg(NgramWmGenerator):
    """ Generate text using LM and Maryland's watermarking method. """
    def __init__(self, 
            *args, 
            gamma: float = 0.25,
            delta: float = 2.0,
            **kwargs
        ):
        super().__init__(*args, **kwargs)        
        self.gamma = gamma
        self.delta = delta
        
    def F_key(self, r, ngram, t):
        """ calculate the watermark key xi at position t """
        bsz = ngram.shape[0]
        batched_xi = []
        for i in range(bsz):
            seed = self.get_seed_rng(ngram[i])
            self.rng.manual_seed(seed)
            # generate a permutation on vocabulary
            vocab_permutation = torch.randperm(self.vocab_size, generator=self.rng)
            greenlist = vocab_permutation[:int(self.gamma * self.vocab_size)] # gamma * n
            xi = torch.zeros(self.vocab_size) # n
            xi[greenlist] = 1
            xi = xi.roll(-r[i])
            batched_xi.append(xi)
        
        return batched_xi  
    
    def Gamma(
        self,
        xis, # a list of 'bsz' xis
        logits: torch.FloatTensor, # (bsz, vocab_size): logits for last token
        temperature: float = 1, # temperature for sampling
        top_p: float = 1, # top p for sampling
    ) -> torch.LongTensor:
        
        modified_logits = logits + self.delta*torch.stack(xis, dim=0).to(device)
        if temperature > 0:
            probs = self.top_p(modified_logits, temperature, top_p)  
            next_token = torch.multinomial(probs, num_samples=1) # one hot of next token, ordered by original probs
        else:
            next_token = torch.argmax(modified_logits, dim=-1)
        next_token = next_token.reshape(-1)
        return next_token

class MarylandGeneratorGseq(GseqWmGenerator):
    """ Generate text using LM and Maryland's watermarking method. """
    def __init__(self, 
            *args, 
            gamma: float = 0.25,
            delta: float = 2.0,
            **kwargs
        ):
        super().__init__(*args, **kwargs)        
        self.gamma = gamma
        self.delta = delta
        self.xis = []
        for _ in range(self.wmkey_len):
            vocab_permutation = torch.randperm(self.vocab_size, generator=self.rng)
            greenlist = vocab_permutation[:int(self.gamma * self.vocab_size)] # gamma * n
            xi = torch.zeros(self.vocab_size) # n
            xi[greenlist] = 1
            self.xis.append(xi)
    
    def Gamma(
        self,
        xis, # a list of 'bsz' xis
        logits: torch.FloatTensor, # (bsz, vocab_size): logits for last token
        temperature: float = 1, # temperature for sampling
        top_p: float = 1, # top p for sampling
    ) -> torch.LongTensor:
        
        modified_logits = logits + self.delta*torch.stack(xis, dim=0).to(device)
        if temperature > 0:
            probs = self.top_p(modified_logits, temperature, top_p)  
            next_token = torch.multinomial(probs, num_samples=1) 
        else:
            next_token = torch.argmax(modified_logits, dim=-1)
        next_token = next_token.reshape(-1)
        return next_token 

class OpenaiGeneratorNg(NgramWmGenerator):
    """ Generate text using LM and Aaronson's watermarking method. """
    def __init__(self, *args, drop_prob=0, tau=0, **kwargs):
        super().__init__(*args, **kwargs)        
        self.drop_prob = drop_prob/100 
        self.tau = tau
    def F_key(self, r, ngram, t):
        """ calculate the watermark key xi at position t """
        bsz = ngram.shape[0]
        batched_xi = []
        for i in range(bsz):
            seed = self.get_seed_rng(ngram[i])
            self.rng.manual_seed(seed)
            # generate xi randomly between [0,1]
            xi = torch.rand(self.vocab_size, generator=self.rng)
            xi = xi.roll(-r[i])
            batched_xi.append(xi)
        
        return batched_xi 
      
    def Gamma(
        self,
        xis, # a list of 'bsz' xis
        logits: torch.FloatTensor, # (bsz, vocab_size): logits for last token
        temperature: float = 1, # temperature for sampling
        top_p: float = 1, # top p for sampling
    ) -> torch.LongTensor:
        if temperature > 0:
            probs = self.top_p(logits, temperature, top_p)  
            xis = torch.stack(xis).to(device)
            if self.tau>0:
                # run exponential soft-minimum sampling
                next_token =  torch.multinomial(torch.softmax((xis.log()/probs)/self.tau, dim=-1), num_samples=1) 
            else:
                # run exponential minimum sampling, drop probability is self.drop_prob
                if np.random.rand()<self.drop_prob:
                    # sample next token based on original probability distribution
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(xis.log()/probs, dim=-1, keepdim=True)
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)
        return next_token

class OpenaiGeneratorGseq(GseqWmGenerator):

    def __init__(self, *args, drop_prob=0, **kwargs):
        super().__init__(*args, **kwargs) 
        self.drop_prob = drop_prob/100
        self.xis = [torch.rand(self.vocab_size, generator=self.rng) for _ in range(self.wmkey_len)]
          
    def Gamma(
        self,
        xis, # a list of 'bsz' xis
        logits: torch.FloatTensor, # (bsz, vocab_size): logits for last token
        temperature: float = 1, # temperature for sampling
        top_p: float = 1, # top p for sampling
    ) -> torch.LongTensor:
        if temperature > 0:
            probs = self.top_p(logits, temperature, top_p)  
            xis = torch.stack(xis).to(device)
            if np.random.rand()<self.drop_prob:
                # sample next token based on original probability distribution
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # sample next token via exponential minimum sampling
                next_token = torch.argmax(xis.log()/probs, dim=-1, keepdim=True)
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)
        return next_token

def Dipmarkprobs_processor(probs, xis, alpha):
    """Process logits to mask out words in greenlist."""
    # Create a batch of permutations
    vocab_permutations = torch.stack(xis).to(device)
    permuted_probs1 = probs.gather(1, vocab_permutations)
    permuted_probs2 = probs.gather(1, vocab_permutations)
    # Calculate cumulative sum for each batch
    probs_sum = torch.cumsum(permuted_probs1, dim=-1)
    # Apply masks
    mask1 = probs_sum <= alpha
    mask2 = probs_sum <= 1 - alpha
    permuted_probs1[mask1] = 0.0
    permuted_probs2[mask2] = 0.0
    permuted_probs = permuted_probs1 + permuted_probs2
    # Normalize probabilities
    permuted_probs.div_(permuted_probs.sum(dim=-1, keepdim=True))
    # Scatter the normalized probabilities back to their original positions
    next_token_probs = torch.zeros_like(probs).to(device).scatter_(1, vocab_permutations, permuted_probs)
    return next_token_probs

class DipmarkGeneratorNg(NgramWmGenerator):
    """ Generate text using LM and DipMark's watermarking method. """
    def __init__(self, 
            *args, 
            alpha: float = 0.45,
            **kwargs
        ):
        super().__init__(*args, **kwargs) 
        self.alpha = alpha
    
    def F_key(self, r, ngram, t):
        """ calculate the watermark key xi at position t """
        bsz = ngram.shape[0]
        batched_xi = []
        for i in range(bsz):
            seed = self.get_seed_rng(ngram[i])
            self.rng.manual_seed(seed)
            xi = torch.randperm(self.vocab_size, generator=self.rng)
            xi = xi.roll(-r[i])
            batched_xi.append(xi)
        return batched_xi 
     
    def Gamma(
        self,
        xis,  # a list of 'bsz' xis
        logits: torch.FloatTensor, # (bsz, vocab_size): logits for last token
        temperature: float = 1, # temperature for sampling
        top_p: float = 1, # top p for sampling
    ) -> torch.LongTensor:
        """
        - partition the permuted vocabrary in two different manners
        - 1.Zeros the probabilities for some tokens at start, where the original probability for these tokens sum to alpha
        - 2.Zeros the probabilities for some tokens at start, where the original probability for these tokens sum to 1-alpha
        - add the above two vectors to a new vector, and sample from this new vector
        """
        if temperature > 0:
            probs = self.top_p(logits, temperature, top_p)   
            probs = Dipmarkprobs_processor(probs, xis, self.alpha)
            next_token = torch.multinomial(probs, num_samples=1) # one hot of next token, ordered by vocab
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)
        return next_token
    
class DipmarkGeneratorGseq(GseqWmGenerator):
    """ Generate text using LM and DipMark's watermarking method. """
    def __init__(self, 
            *args, 
            alpha: float = 0.45,
            **kwargs
        ):
        super().__init__(*args, **kwargs) 
        self.alpha = alpha
        self.xis = [torch.randperm(self.vocab_size, generator=self.rng) for _ in range(self.wmkey_len)]

    def Gamma(
        self,
        xis,  # a list of 'bsz' xis
        logits: torch.FloatTensor, # (bsz, vocab_size): logits for last token
        temperature: float = 1, # temperature for sampling
        top_p: float = 1, # top p for sampling
    ) -> torch.LongTensor:
        if temperature > 0:
            probs = self.top_p(logits, temperature, top_p)   
            probs = Dipmarkprobs_processor(probs, xis, self.alpha)
            next_token = torch.multinomial(probs, num_samples=1) # one hot of next token, ordered by vocab
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)
        return next_token

def inv_gumbel_cdf(xi, mu=0, beta=1, eps=1e-20):
    return mu - beta * torch.log(-torch.log(xi + eps))

class GumbelSoftGeneratorNg(NgramWmGenerator):
    """ Generate text using LM and Gumbel-softmax watermarking method. """
    def __init__(self, 
            *args, 
            drop_prob = 0,
            tau=0,
            **kwargs,
        ):
        super().__init__(*args, **kwargs) 
        self.drop_prob = drop_prob/100
        self.tau=tau
        
    def F_key(self, r, ngram, t):
        """ calculate the watermark key xi at position t """
        bsz = ngram.shape[0]
        batched_xi = []
        for i in range(bsz):
            seed = self.get_seed_rng(ngram[i])
            self.rng.manual_seed(seed)
            xi = torch.rand(self.vocab_size, generator=self.rng)
            xi = inv_gumbel_cdf(xi)
            xi = xi.roll(-r[i])
            batched_xi.append(xi)
            
        return batched_xi 
     
    def Gamma(
        self,
        xis,  # a list of 'bsz' xis
        logits: torch.FloatTensor, # (bsz, vocab_size): logits for last token
        temperature: float = 1, # temperature for sampling
        top_p: float = 1, # top p for sampling
    ) -> torch.LongTensor:  
        if temperature > 0:
            probs = self.top_p(logits, temperature, top_p)
            xis = torch.stack(xis).to(device)
            if self.tau>0:
                # run Gumbel-softmax
                next_token =  torch.multinomial(torch.softmax((probs.log()+xis)/self.tau, dim=-1), num_samples=1) 
            else:
                # run Logp-Addition, drop probability is self.drop_prob
                if np.random.rand()<self.drop_prob:
                    # sample next token based on original probability distribution
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(probs.log()+xis, dim=-1, keepdim=True)
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)
        return next_token

class GumbelSoftGeneratorGseq(GseqWmGenerator):
    """ Generate text using LM and Gumbel-softmax watermarking method. """
    def __init__(self, 
            *args, 
            **kwargs
        ):
        super().__init__(*args, **kwargs) 
        self.xis = [inv_gumbel_cdf(torch.rand(self.vocab_size, generator=self.rng)) for _ in range(self.wmkey_len)]
     
    def Gamma(
        self,
        xis,  # a list of 'bsz' xis
        logits: torch.FloatTensor, # (bsz, vocab_size): logits for last token
        temperature: float = 0.1, # temperature for sampling
        top_p: float = 1, # top p for sampling
    ) -> torch.LongTensor:  
        if temperature > 0:
            xis = torch.stack(xis).to(device) #(bsz, vocab_size)
            probs = self.top_p(logits+xis, temperature, top_p)   
            next_token = torch.multinomial(probs, num_samples=1) # one hot of next token, ordered by vocab
        else:
            next_token = torch.argmax(xis+logits, dim=-1)     
        next_token = next_token.reshape(-1)
        return next_token

class ITSGeneratorNg(NgramWmGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) 
    
    def F_key(self, r, ngram, t):
        """ calculate the watermark key xi at position t """
        bsz = ngram.shape[0]
        batched_xi = []
        for i in range(bsz):
            seed = self.get_seed_rng(ngram[i])
            self.rng.manual_seed(seed)
            sigma = torch.rand(1, generator=self.rng)
            pi = torch.randperm(self.vocab_size, generator=self.rng)
            pi = pi.roll(-r[i])
            batched_xi.append((sigma, pi))
  
        return batched_xi 

    def Gamma(
        self,
        xis,
        logits: torch.FloatTensor, # (bsz, vocab_size): logits for last token
        temperature: float = 1, # temperature for sampling
        top_p: float = 1, # top p for sampling
    ) -> torch.LongTensor:
        """
        Decode next token by logits and self.xis[wmkey_idx]
        we implement ITS(inverse transform sampling) here.
        """
        sigmas = []
        pis = []
        for xi in xis:
            sigma, pi = xi
            sigmas.append(sigma)
            pis.append(pi)
        sigmas=torch.tensor(sigmas).to(device)
        pis = torch.stack(pis).to(device)
        if temperature > 0:
            probs = self.top_p(logits, temperature, top_p)  
            inverse_pi = torch.argsort(pis, dim=-1)
            probs = probs.gather(-1, inverse_pi)
            probs_sum = torch.cumsum(probs, dim=-1)
            mask = (probs_sum >= sigmas.reshape(len(xis), 1)).to(dtype=torch.int)
            next_token = inverse_pi.gather(-1, mask.argmax(dim=-1, keepdim=True))
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)
        return next_token

class ITSGeneratorGseq(GseqWmGenerator):
    """ Generate text using LM and stanford's watermarking method(ITS-Inverse transform sampling). """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) 
        self.sigmas = torch.rand(self.wmkey_len, generator=self.rng)
        self.pis = [torch.randperm(self.vocab_size, generator=self.rng) for _ in range(self.wmkey_len)]
        self.xis = [(self.sigmas[i], self.pis[i]) for i in range(self.wmkey_len)]

    def Gamma(
        self,
        xis,
        logits: torch.FloatTensor, # (bsz, vocab_size): logits for last token
        temperature: float = 1, # temperature for sampling
        top_p: float = 1, # top p for sampling
    ) -> torch.LongTensor:
        """
        Decode next token by logits and self.xis[wmkey_idx]
        we implement ITS(inverse transform sampling) here.
        """
        sigmas = []
        pis = []
        for xi in xis:
            sigma, pi = xi
            sigmas.append(sigma)
            pis.append(pi)
        sigmas=torch.Tensor(sigmas).to(device)
        pis = torch.stack(pis).to(device)
        if temperature > 0:
            probs = self.top_p(logits, temperature, top_p)  
            inverse_pi = torch.argsort(pis, dim=-1)
            probs = probs.gather(-1, inverse_pi)
            probs_sum = torch.cumsum(probs, dim=-1)
            mask = (probs_sum >= sigmas.reshape(len(xis), 1)).to(dtype=torch.int)
            next_token = inverse_pi.gather(-1, mask.argmax(dim=-1, keepdim=True))
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)
        return next_token


