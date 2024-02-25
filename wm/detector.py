# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Dict, Callable
import numpy as np
import torch
from scipy import special
from transformers import LlamaTokenizer
from wm.utils import sample_texts

class WmDetector:
    def __init__(self, 
            tokenizer: LlamaTokenizer,
            seed: int = 42,
            shift_max: int = 0):
        
        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size
        self.seed = seed
        self.shift_max = shift_max
        self.rng = torch.Generator(device='cpu')

    def get_szp_by_t(self, text:str, eps=1e-200):
        """
        Get p-value for each text.
        Args:
            text: a str, the text to detect
        Output:
            shift, zscore, pvalue, ntokens
        """
        pass 

class NgramWmDetector(WmDetector):
    def __init__(self, 
            tokenizer: LlamaTokenizer, 
            seed: int = 42,
            shift_max: int = 0,
            ngram: int = 1,
            seeding: str = 'hash',
            hash_key: int = 35317,
            scoring_method: str = "none",
        ):
        super().__init__(tokenizer, seed, shift_max)
        # watermark config
        self.ngram = ngram
        self.seeding = seeding
        self.hash_key = hash_key
        self.hashtable = np.random.permutation(1000003)
        self.scoring_method = scoring_method

    def hashint(self, integer_array):
        """Adapted from https://github.com/jwkirchenbauer/lm-watermarking"""
        return self.hashtable[integer_array % len(self.hashtable)] 
    
    def get_seed_rng(self, input_ids: List[int]):
        """
        Seed RNG with hash of input_ids.
        Adapted from https://github.com/jwkirchenbauer/lm-watermarking
        """
        if self.seeding == 'hash':
            seed = self.seed
            for i in input_ids:
                seed = (seed * self.hash_key + i) % (2 ** 64 - 1)
        elif self.seeding == 'additive':
            seed = self.hash_key * np.sum(input_ids)
            seed = self.hashint(seed)
        elif self.seeding == 'skip':
            seed = self.hash_key * input_ids[0]
            seed = self.hashint(seed)
        elif self.seeding == 'min':
            seed = self.hashint(self.hash_key * input_ids)
            seed = np.min(seed)
        return seed

    def aggregate_scores(self, scores: List[np.array], aggregation: str = 'mean'):
        """Aggregate scores along a text."""
        scores = np.asarray(scores) # seq_len * (shift_max+1)
        if aggregation == 'sum':
            return scores.sum(axis=0)
        elif aggregation == 'mean':
            return scores.mean(axis=0) if scores.shape[0]!=0 else np.ones(shape=(self.vocab_size))
        elif aggregation == 'max':
            return scores.max(axis=0)
        else:
            raise ValueError(f'Aggregation {aggregation} not supported.')

    def get_scores_by_t(self, text: str, toks: int):
        """
        Get score increment for each token in a text
        Args:
            text: a text
            scoring_method: 
                'none': score all ngrams
                'v1': only score toksns for which wm window is unique
                'v2': only score unique {wm window+tok} is unique
            ntoks_max: maximum number of tokens
        Output:
            score: [np array of score increments for every token and payload] for a text
        """
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        total_len = len(token_ids) if toks is None else min(len(token_ids), toks+4)
        start_pos = self.ngram +1
        rts = []
        seen_ntuples = set()
        for cur_pos in range(start_pos, total_len):
            ngram_tokens = token_ids[cur_pos-self.ngram:cur_pos] 
            if self.scoring_method == 'v1':
                tup_for_unique = tuple(ngram_tokens)
                if tup_for_unique in seen_ntuples:
                    continue
                seen_ntuples.add(tup_for_unique)
            elif self.scoring_method == 'v2':
                tup_for_unique = tuple(ngram_tokens + token_ids[cur_pos:cur_pos+1])
                if tup_for_unique in seen_ntuples:
                    continue
                seen_ntuples.add(tup_for_unique)
            rt = self.score_tok(ngram_tokens, token_ids[cur_pos]) 
            rt = rt[:self.shift_max+1]
            rts.append(rt)  
        return rts

    def get_szp_by_t(self, text: str, toks=None, eps=1e-200):
        ptok_scores = self.get_scores_by_t(text, toks)
        ptok_scores = np.asarray(ptok_scores) # ntoks x (shift_max+1)
        ntoks = ptok_scores.shape[0]
        aggregated_scores = ptok_scores.sum(axis=0) if ntoks!=0 else np.zeros(shape=ptok_scores.shape[-1]) # shift_max+1
        pvalues = [self.get_pvalue(score, ntoks, eps=eps) for score in aggregated_scores] # shift_max+1
        zscores = [self.get_zscore(score, ntoks) for score in aggregated_scores] # shift_max+1
        pvalue = min(pvalues)
        shift = pvalues.index(pvalue)
        zscore = zscores[shift]
        return int(shift), float(zscore), float(pvalue), ntoks
    
    def score_tok(self, ngram_tokens: List[int], token_id: int):
        """ for each token in the text, compute the unit score """
        raise NotImplementedError
    
    def get_zscore(self, score: int, ntoks: int):
        """ compute the zscore from the total score and the number of tokens """
        raise NotImplementedError
    
    def get_pvalue(self, score: int, ntoks: int, eps: float=1e-200):
        """ compute the p-value from the total score and the number of tokens """
        raise NotImplementedError

class GseqWmDetector(WmDetector):
    """ This follows the procedure proposed by stanford's paper https://arxiv.org/abs/2307.15593 """
    def __init__(self, 
            tokenizer: LlamaTokenizer,
            seed: int = 42,
            shift_max: int = 0,
            wmkey_len: int = 256,
        ):
        super().__init__(tokenizer, seed, shift_max)
        self.wmkey_len = wmkey_len
        self.xis = []

    def get_szp_by_t(self, text, toks=None, eps=1e-200):
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if toks is not None:
            token_ids = token_ids[:toks]
        m = len(token_ids)
        n = len(self.xis)
        seq_scores = [self.seq_score(token_ids, self.xis[shift]) for shift in range(self.shift_max+1)]
        pvalues = [self.get_pvalue(score, m, eps) for score in seq_scores]
        zscores = [self.get_zscore(score, m) for score in seq_scores]
        pvalue = min(pvalues)
        shift = pvalues.index(pvalue)
        zscore = zscores[shift]

        return int(shift), float(zscore), float(pvalue), m

    def seq_score(self, token_ids: List[int], xi):
        scores = 0
        for token_id in token_ids:
            scores += self.unit_score(xi, token_id)
        return scores
    
    def get_zscore(self, score: int, ntoks: int):
        """ compute the zscore from the total score and the number of tokens """
        raise NotImplementedError
    
    def get_pvalue(self, score: int, ntoks: int, eps: float=1e-200):
        """ compute the p-value from the total score and the number of tokens """
        raise NotImplementedError

    def unit_score(self, xi, token_id):
        raise NotImplementedError

class MarylandDetectorNg(NgramWmDetector):
    def __init__(self, *args, gamma: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
    
    def score_tok(self, ngram_tokens, token_id):
        """ 
        score_t = 1 if token_id in greenlist else 0 
        The last line shifts the scores by token_id. 
        ex: scores[0] = 1 if token_id in greenlist else 0
            scores[1] = 1 if token_id in (greenlist shifted of 1) else 0
            ...
        The score for each shift will be given by scores[shift]
        """
        seed = self.get_seed_rng(ngram_tokens)
        self.rng.manual_seed(seed)
        vocab_permutation = torch.randperm(self.vocab_size, generator=self.rng).numpy()
        greenlist = vocab_permutation[:int(self.gamma * self.vocab_size)] # gamma * n toks in the greenlist
        xi = np.zeros(self.vocab_size)
        xi[greenlist] = 1 
        return np.roll(xi, -token_id)
    
    def get_zscore(self, score, ntoks):
        zscore = (score - self.gamma * ntoks) / np.sqrt(self.gamma * (1 - self.gamma) * ntoks)
        return zscore 
               
    def get_pvalue(self, score, ntoks, eps=1e-200):
        """ from cdf of a normal distribution """
        zscore = self.get_zscore(score, ntoks)
        pvalue = 0.5 * special.erfc(zscore / np.sqrt(2))
        return max(pvalue, eps)

class MarylandDetectorGseq(GseqWmDetector):
    def __init__(self, *args, gamma: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.xis = []
        for _ in range(self.wmkey_len):
            vocab_permutation = torch.randperm(self.vocab_size, generator=self.rng).numpy()
            greenlist = vocab_permutation[:int(self.gamma * self.vocab_size)] # gamma * n
            xi = np.zeros(self.vocab_size)# n
            xi[greenlist] = 1
            self.xis.append(xi)
    
    def get_zscore(self, score, ntoks):
        zscore = (score - self.gamma * ntoks) / np.sqrt(self.gamma * (1 - self.gamma) * ntoks)
        return zscore 
               
    def get_pvalue(self, score, ntoks, eps=1e-200):
        """ from cdf of a normal distribution """
        zscore = self.get_zscore(score, ntoks)
        pvalue = 0.5 * special.erfc(zscore / np.sqrt(2))
        return max(pvalue, eps)
    
    def unit_score(self, xi, token_id):
        return xi[token_id] 
    
class OpenaiDetectorNg(NgramWmDetector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def score_tok(self, ngram_tokens, token_id):
        """ 
        score_t = -log(1 - rt[token_id]])
        The last line shifts the scores by token_id. 
        ex: scores[0] = r_t[token_id]
            scores[1] = (r_t shifted of 1)[token_id]
            ...
        The score for each shift will be given by scores[shift]
        """
        seed = self.get_seed_rng(ngram_tokens)
        self.rng.manual_seed(seed)
        xi = torch.rand(self.vocab_size, generator=self.rng).numpy()
        scores = np.roll(-np.log(1 - xi), -token_id)
        return scores
    
    def get_zscore(self, score, ntoks):
        mu0 = 1
        sigma0 = 1
        zscore = (score/ntoks - mu0) / (sigma0 / np.sqrt(ntoks))
        return zscore
    
    def get_pvalue(self, score, ntoks, eps=1e-200):
        """ from cdf of a normal distribution """
        zscore = self.get_zscore(score, ntoks)
        pvalue = 0.5 * special.erfc(zscore / np.sqrt(2))
        return max(pvalue, eps)
    

class OpenaiDetectorGseq(GseqWmDetector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.xis = [torch.rand(self.vocab_size, generator=self.rng).numpy() for _ in range(self.wmkey_len)]
        
    def get_zscore(self, score, ntoks):
        mu0 = 1
        sigma0 = 1
        zscore = (score/ntoks - mu0) / (sigma0 / np.sqrt(ntoks))
        return zscore
    
    def get_pvalue(self, score, ntoks, eps=1e-200):
        """ from cdf of a normal distribution """
        zscore = self.get_zscore(score, ntoks)
        pvalue = 0.5 * special.erfc(zscore / np.sqrt(2))
        return max(pvalue, eps)
    
    def unit_score(self, xi, token_id):
        return -np.log(1 - xi)[token_id]

class DipmarkDetectorNg(NgramWmDetector):
    def __init__(self, *args, gamma: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        
    def score_tok(self, ngram_tokens, token_id):
        """ 
        score_t = 1 if token_id in greenlist else 0 
        The score for each payload will be given by scores[payload]
        """
        seed = self.get_seed_rng(ngram_tokens)
        self.rng.manual_seed(seed)
        vocab_permutation = torch.randperm(self.vocab_size, generator=self.rng).numpy().tolist()
        idx = vocab_permutation.index(token_id)
        shift_range = np.arange(self.shift_max+1)
        scores = (idx-shift_range+self.vocab_size)%self.vocab_size >= self.gamma*self.vocab_size
        return scores 
    
    def get_zscore(self, score, ntoks):
        zscore = (score - (1-self.gamma)*ntoks)/np.sqrt(ntoks)
        return zscore
           
    def get_pvalue(self, score, ntoks, eps=1e-200):
        """ This is z-score statistic , not pvalue. But is still satisfy the property:
        For watermarked text, return a small value,
        For unwatermarked text, return a big value.
        We use this statistic because that paper suggests this statistic.
        """
        pseudo_pvalue = 1 - self.get_zscore(score, ntoks)
        return pseudo_pvalue

class DipmarkDetectorGseq(GseqWmDetector):
    def __init__(self, *args, gamma: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.xis = [torch.randperm(self.vocab_size, generator=self.rng).numpy().tolist() for _ in range(self.wmkey_len)]

    def get_zscore(self, score, ntoks):
        zscore = (score - (1-self.gamma)*ntoks)/np.sqrt(ntoks)
        return zscore
           
    def get_pvalue(self, score, ntoks, eps=1e-200):
        """ This is z-score statistic , not pvalue. But is still satisfy the property:
        For watermarked text, return a small value,
        For unwatermarked text, return a big value.
        We use this statistic because that paper suggests this statistic.
        """
        pseudo_pvalue = 1 - self.get_zscore(score, ntoks)
        return pseudo_pvalue
    
    def unit_score(self, xi, token_id):
        idx = xi.index(token_id)
        return idx >= self.gamma*self.vocab_size
    
def inv_gumbel_cdf(xi, mu=0, beta=1, eps=1e-20):
    return mu - beta * np.log(-np.log(xi + eps))

class GumbelSoftDetectorNg(NgramWmDetector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def score_tok(self, ngram_tokens, token_id):
        seed = self.get_seed_rng(ngram_tokens)
        self.rng.manual_seed(seed)
        xi = torch.rand(self.vocab_size, generator=self.rng).numpy()
        xi = inv_gumbel_cdf(xi)
        scores = np.roll(xi, -token_id)
        return scores
    
    def get_zscore(self, score, ntoks):
        mu = 0.57721
        sigma = np.pi / np.sqrt(6)
        zscore = (score/ntoks - mu) / (sigma / np.sqrt(ntoks))
        return zscore
    
    def get_pvalue(self, score, ntoks, eps=1e-200):
        """ from cdf of a normal distribution """
        zscore = self.get_zscore(score, ntoks)
        pvalue = 0.5 * special.erfc(zscore / np.sqrt(2))
        return max(pvalue, eps)
    
class GumbelSoftDetectorGseq(GseqWmDetector):
    def __init__(self, 
            *args, 
            **kwargs
        ):
        super().__init__(*args, **kwargs) 
        self.xis = [inv_gumbel_cdf(torch.rand(self.vocab_size, generator=self.rng).numpy()) for _ in range(self.wmkey_len)]

    def get_zscore(self, score, ntoks):
        mu = 0.57721
        sigma = np.pi / np.sqrt(6)
        zscore = (score/ntoks - mu) / (sigma / np.sqrt(ntoks))
        return zscore
    
    def get_pvalue(self, score, ntoks, eps=1e-200):
        """ from cdf of a normal distribution """
        zscore = self.get_zscore(score, ntoks)
        pvalue = 0.5 * special.erfc(zscore / np.sqrt(2))
        return max(pvalue, eps)
    
    def unit_score(self, xi, token_id):
        return xi[token_id]

class ITS_helper:
    def __init__(self, tokenizer, ref_count, natural_text_path, max_gen_len, wmkey_len):
        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size
        self.ref_count = ref_count
        self.natural_text_path = natural_text_path
        self.max_gen_len = max_gen_len
        self.wmkey_len = wmkey_len
     
    def construct_ref_dist(self):
        texts = sample_texts(self.ref_count, self.natural_text_path)
        xis = self.sample_xis()
        ref_dist = []
        for text in texts:
            y = self.tokenizer.encode(text, add_special_tokens=False)
            # y is irrelevant to xis, so the test_statistic should be high
            ref_dist.append(self.test_statistic(y[:self.max_gen_len], xis))
        return ref_dist
    
    def floor(self, i):
        return i/(self.vocab_size-1)
    
    def test_statistic(self, token_ids, xis):
        ntoks = len(token_ids)
        scores = 0.0
        for xi, token_id in zip(xis[:ntoks], token_ids):
            sigma, pi = xi
            scores += np.abs(sigma - self.floor(pi[token_id]))
        return scores/ntoks

    def sample_xis(self):
        sigmas = np.random.rand(self.wmkey_len)
        pis = [np.random.permutation(self.vocab_size) for _ in range(self.wmkey_len)]
        xis = [(sigmas[i],pis[i]) for i in range(self.wmkey_len)]
        return xis
    
class ITSDetectorNg(NgramWmDetector):
    def __init__(self, *args, ref_count, natural_text_path, max_gen_len, wmkey_len, **kwargs):
        super().__init__(*args, **kwargs)
        self.ITS_helper = ITS_helper(self.tokenizer, ref_count, natural_text_path, max_gen_len, wmkey_len)
        self.ref_dist = self.ITS_helper.construct_ref_dist()

    def floor(self, i):
        return i/(self.vocab_size-1)
    
    def score_tok(self, ngram_tokens, token_id):
        seed = self.get_seed_rng(ngram_tokens)
        self.rng.manual_seed(seed)
        sigma = torch.rand(1, generator=self.rng).numpy()
        pi = torch.randperm(self.vocab_size, generator=self.rng).numpy()
        scores = np.abs(sigma - self.floor(pi))
        scores = np.roll(scores, -token_id)
        return scores
    
    def get_zscore(self, score, ntoks, eps=1e-200):
        """ compare the score with other randomly sampled scores """
        count = 0
        for ref in self.ref_dist:
            count += ref <= score/ntoks
            
        return 1-(count+1.0)/(len(self.ref_dist)+1)
    
    def get_pvalue(self, score, ntoks, eps=1e-200):
        p_val = 1-self.get_zscore(score, ntoks)
        return max(p_val, eps)
    
class ITSDetectorGseq(GseqWmDetector):
    '''This class implements ITS(inverse transform sampling) detector's details'''
    def __init__(self, *args, ref_count, natural_text_path, max_gen_len, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigmas = torch.rand(self.wmkey_len, generator=self.rng).numpy()
        self.pis = [torch.randperm(self.vocab_size, generator=self.rng).numpy() for _ in range(self.wmkey_len)]
        self.xis = [(self.sigmas[i],self.pis[i]) for i in range(self.wmkey_len)]
        self.ITS_helper = ITS_helper(self.tokenizer, ref_count, natural_text_path, max_gen_len, self.wmkey_len)
        self.ref_dist = self.ITS_helper.construct_ref_dist()

    def get_zscore(self, score, ntoks, eps=1e-200):
        """ compare the score with other randomly sampled scores """
        count = 0
        for ref in self.ref_dist:
            count += ref <= score/ntoks
            
        return 1-(count+1.0)/(len(self.ref_dist)+1)
    
    def get_pvalue(self, score, ntoks, eps=1e-200):
        p_val = 1-self.get_zscore(score, ntoks)
        return max(p_val, eps)
    
    def floor(self, i):
        return i/(self.vocab_size-1)
    
    def unit_score(self, xi, token_id):
        sigma, pi = xi
        score = np.abs(sigma - self.floor(pi[token_id]))
        return score
    

    