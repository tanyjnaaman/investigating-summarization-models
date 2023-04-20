from typing import List, Dict
import torch
from torchmetrics.functional.text.rouge import rouge_score

def evaluate_rouge_score(
        preds: List[str], 
        target: List[str], 
        accumulate='best', 
        use_stemmer=False, 
        normalizer=None, 
        tokenizer=None, 
        rouge_keys=('rouge1', 'rouge2', 'rougeL', 'rougeLsum',)
) -> Dict[str, torch.Tensor]: 
    """
    See https://torchmetrics.readthedocs.io/en/stable/text/rouge_score.html.
    """
    return rouge_score(preds, target, accumulate, use_stemmer, normalizer, tokenizer, rouge_keys)


