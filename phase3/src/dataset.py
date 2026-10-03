import json
import torch
from torch.utils.data import Dataset
import sentencepiece as spm
import numpy as np

class Phase3ReasoningDataset(Dataset):
    """
    JSONL Dataset loader for Phase 3 Reasoning Fine-tuning.
    Implements answer-only loss masking by replacing prompt targets with -100.
    """
    def __init__(self, jsonl_path, tokenizer_path, max_length=512):
        self.max_length = max_length
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(str(tokenizer_path))
        
        self.examples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                self.examples.append(json.loads(line))
                
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        d = self.examples[idx]
        # Exact format from generator: 
        # "प्रश्न: <premises> <question> उत्तर: <answer>"
        # We need to construct the prompt up to and including the space after "उत्तर:"
        # Wait, the generator output has: 'premise', 'question', 'answer'
        # In generator logic: f"प्रश्न: {d['premise']} {d['question']} उत्तर: {d['answer']}"
        
        prompt_text = f"प्रश्न: {d['premise']} {d['question']} उत्तर: "
        full_text = prompt_text + str(d['answer'])
        
        prompt_ids = self.sp.Encode(prompt_text, out_type=int)
        full_ids = self.sp.Encode(full_text, out_type=int)
        
        # Append EOS
        full_ids.append(self.sp.eos_id())
        
        # Truncate if necessary (rare for this dataset)
        if len(full_ids) > self.max_length + 1:
            full_ids = full_ids[:self.max_length + 1]
            
        # The input x is everything except the last token
        x = full_ids[:-1]
        
        # The target y is everything except the first token
        y = full_ids[1:]
        
        # We want to mask loss on all tokens up to the end of the prompt.
        # Length of prompt_ids is L.
        # The first token to predict that bears loss is the first token of the answer.
        # It is located at full_ids[L].
        # In the shifted target array `y`, the element at index L-1 is full_ids[L].
        # Therefore, indices 0 through L-2 in `y` should be masked.
        
        L = len(prompt_ids)
        
        y_masked = list(y)
        for i in range(L - 1):
            if i < len(y_masked):
                y_masked[i] = -100
                
        # Pad to max_length
        pad_len = self.max_length - len(x)
        if pad_len > 0:
            # use 0 for pad in x, -100 for pad in y
            x = x + [0] * pad_len
            y_masked = y_masked + [-100] * pad_len
            
        return torch.tensor(x, dtype=torch.long), torch.tensor(y_masked, dtype=torch.long)


def run_unit_test():
    """
    Deterministic unit test for loss mask alignment.
    """
    import os
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    
    # We will use the Hindi tokenizer for the test.
    tokenizer_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../hindi/tokenizer/models/hindi_bpe_10k_final.model'))
    if not os.path.exists(tokenizer_path):
        print(f"Skipping dataset unit test, tokenizer not found at {tokenizer_path}")
        return
        
    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_path)
    
    d = {
        'premise': 'ए बी से बड़ा है।',
        'question': 'सबसे बड़ा कौन है?',
        'answer': 'ए'
    }
    
    prompt_text = f"प्रश्न: {d['premise']} {d['question']} उत्तर: "
    full_text = prompt_text + d['answer']
    
    prompt_ids = sp.Encode(prompt_text, out_type=int)
    full_ids = sp.Encode(full_text, out_type=int)
    full_ids.append(sp.eos_id())
    
    x = full_ids[:-1]
    y_raw = full_ids[1:]
    
    L = len(prompt_ids)
    y_masked = list(y_raw)
    for i in range(L - 1):
        y_masked[i] = -100
        
    print("\n--- LOSS MASK ALIGNMENT UNIT TEST ---\n")
    print("DECODED INPUT       :", sp.Decode(x))
    print("TOKEN IDS           :", full_ids)
    print("DECODED PROMPT      :", sp.Decode(prompt_ids))
    print("PROMPT TOKEN LENGTH :", L)
    print("LABELS (UNSHIFTED)  :", full_ids)
    print("LABELS (SHIFTED)    :", y_raw)
    print("LABELS (MASKED)     :", y_masked)
    
    # Analyze exactly what is contributing to loss
    print("\nPOSITIONS CONTRIBUTING TO LOSS:")
    for i, target_id in enumerate(y_masked):
        if target_id != -100:
            # The token we are given to predict this is x[i]
            given_tokens = x[:i+1]
            given_text = sp.Decode(given_tokens)
            target_str = sp.IdToPiece(target_id) if target_id != sp.eos_id() else "<EOS>"
            print(f"Index {i:2d} | Given: {given_text!r} | Target: {target_str} (ID: {target_id})")

    # Assertions
    assert y_masked[L-2] == -100, "Last prompt token target should be masked"
    assert y_masked[L-1] != -100, "First answer token target should NOT be masked"
    assert y_masked[-1] == sp.eos_id(), "EOS should be the last valid target"
    print("\n[PASS] Loss mask assertions passed.")

if __name__ == '__main__':
    run_unit_test()
