"""
Phase 3 — Full Dataset Validator
==================================
Validates every JSONL example for scale generation (24k total).
Produces exact validation results and extensive distribution reports.
"""

import json
import os
import re
import sys
import statistics
from collections import Counter, defaultdict

import sentencepiece as spm

# Add Phase 3 src to path for solver
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from reasoning_solvers import ReasoningSolver

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUTPUT_DIR   = 'results/phase3'
HINDI_MODEL  = 'hindi/tokenizer/models/hindi_bpe_10k_final.model'
NEPALI_MODEL = 'nepali/tokenizer/models/nepali_bpe_10k_assignment.model'
MAX_TOKENS   = 512

EXPECTED_COUNTS = {'train': 9600, 'val': 1200, 'test': 1200}

REQUIRED_FIELDS = {'id', 'language', 'split', 'task_type', 'attribute', 'question_category',
                   'template_id', 'entity_category', 'premise', 'question',
                   'answer', 'entities', 'reasoning_relation', 'unk_count'}

VALID_LANGUAGES   = {'hindi', 'nepali'}
VALID_SPLITS      = {'train', 'val', 'test'}
VALID_TASKS       = {'direct', 'attribute', 'transitive'}
VALID_ATTRIBUTES  = {'height', 'age', 'price', 'quantity', 'generic_order'}

ATTR_ENTITY_COMPAT = {
    'height':        {'people', 'objects'},
    'age':           {'people'},
    'price':         {'objects'},
    'quantity':      {'people', 'objects', 'substances'},
    'generic_order': {'people', 'objects'},
}

QCAT_ENTITY_COMPAT = {
    'possession_count': {'people'},
    'substance_amount': {'substances'},
    'object_count':     {'objects'},
}

SUPERLATIVE_KEYWORDS = {
    'hindi':  ['सबसे', 'सर्वाधिक'],
    'nepali': ['सबैभन्दा'],
}

UNSEEN_TEMPLATE_IDS = {'H3', 'H7', 'H9', 'H22', 'H31', 'H40',
                       'N3', 'N7', 'N9', 'N22', 'N31', 'N40'}

sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

solver = ReasoningSolver()

def check_example(d, sp, errors):
    eid = d.get('id', '?')

    missing = REQUIRED_FIELDS - set(d.keys())
    if missing:
        errors.append(f"[{eid}] Missing fields: {missing}")
        return

    lang  = d['language']
    split = d['split']
    attr  = d['attribute']
    task  = d['task_type']
    ecat  = d['entity_category']
    q_cat = d.get('question_category', attr)
    tid   = d['template_id']

    # Enums
    if lang  not in VALID_LANGUAGES:  errors.append(f"[{eid}] Invalid language: {lang}")
    if split not in VALID_SPLITS:     errors.append(f"[{eid}] Invalid split: {split}")
    if task  not in VALID_TASKS:      errors.append(f"[{eid}] Invalid task_type: {task}")
    if attr  not in VALID_ATTRIBUTES: errors.append(f"[{eid}] Invalid attribute: {attr}")

    # Placeholders
    for field in ('premise', 'question', 'answer'):
        if re.search(r'\{[EV]\d+\}', str(d[field])):
            errors.append(f"[{eid}] Unresolved placeholder in '{field}'")

    # Empty answer
    if not str(d['answer']).strip():
        errors.append(f"[{eid}] Empty answer")

    # Entity compatibility
    if attr in ATTR_ENTITY_COMPAT and ecat not in ATTR_ENTITY_COMPAT[attr]:
        errors.append(f"[{eid}] entity_category '{ecat}' incompatible with attribute '{attr}'")
    if attr == 'quantity' and q_cat in QCAT_ENTITY_COMPAT and ecat not in QCAT_ENTITY_COMPAT[q_cat]:
        errors.append(f"[{eid}] entity_category '{ecat}' incompatible with question_category '{q_cat}'")

    # Leakage
    kws = SUPERLATIVE_KEYWORDS.get(lang, [])
    ans_str = str(d['answer'])
    for kw in kws:
        if kw in d['premise'] and ans_str in d['premise']:
            errors.append(f"[{eid}] Possible answer leakage: '{ans_str}' + '{kw}' in premise")

    # Template mapping compatibility
    if '+' in tid: tids = tid.split('+')
    else: tids = [tid]

    from generate_reasoning_data import HINDI_TEMPLATES, NEPALI_TEMPLATES
    tpl_map = HINDI_TEMPLATES if lang == 'hindi' else NEPALI_TEMPLATES
    for t in tids:
        if t in tpl_map:
            tpl_attr = tpl_map[t]['attribute']
            if tpl_attr != attr:
                errors.append(f"[{eid}] Template {t} attr '{tpl_attr}' != sample '{attr}'")
            tpl_qcat = tpl_map[t].get('question_category', tpl_attr)
            if tpl_qcat != q_cat:
                errors.append(f"[{eid}] Template {t} q_cat '{tpl_qcat}' != sample '{q_cat}'")

    # Token counts & EOS
    text = f"प्रश्न: {d['premise']} {d['question']} उत्तर: {d['answer']}"
    token_ids = sp.encode(text)
    unks = sum(1 for i in token_ids if sp.is_unknown(i))
    if unks > 0:
        errors.append(f"[{eid}] {unks} UNK token(s)")
    if len(token_ids) > MAX_TOKENS:
        errors.append(f"[{eid}] Token count {len(token_ids)} > {MAX_TOKENS}")
    return len(token_ids)


def get_reasoning_signature(d):
    """Produces a canonical string representing the underlying abstract reasoning instance."""
    rel = d['reasoning_relation']
    if 'attribute' in rel:
        return f"{d['attribute']}::{d['task_type']}::{rel}::{d['entity_category']}::{'-'.join(sorted(d['entities']))}"
    else:
        # Standardize relations by sorting them
        try:
            rel_list = eval(rel)
            if isinstance(rel_list, list):
                s_rels = str(sorted([str(x) for x in rel_list]))
                return f"{d['attribute']}::{d['task_type']}::{s_rels}::{d['entity_category']}"
        except Exception:
            pass
        return f"{d['attribute']}::{d['task_type']}::{rel}::{d['entity_category']}"


def run_validation():
    hindi_sp  = spm.SentencePieceProcessor(model_file=HINDI_MODEL)
    nepali_sp = spm.SentencePieceProcessor(model_file=NEPALI_MODEL)

    stats = {}
    all_errors = []

    for lang in ['hindi', 'nepali']:
        sp = hindi_sp if lang == 'hindi' else nepali_sp
        
        # Cross-split structures per language
        seen_rendered = {}    # text -> split
        seen_reasoning = {}   # signature -> split
        
        for split in ['train', 'val', 'test']:
            path = os.path.join(OUTPUT_DIR, f'{lang}_{split}.jsonl')
            if not os.path.exists(path):
                print(f"[FAIL] Missing {path}")
                sys.exit(1)
                
            dist = {
                'tasks': Counter(), 'attrs': Counter(), 'qcats': Counter(),
                'tpls': Counter(), 'entities': Counter(), 'ans': Counter(),
                'chain_lens': Counter()
            }
            token_lengths = []
            test_familiar, test_heldout = 0, 0
            count = 0
            
            with open(path, encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    d = json.loads(line)
                    count += 1
                    
                    tlen = check_example(d, sp, all_errors)
                    if tlen: token_lengths.append(tlen)
                        
                    # Stats
                    dist['tasks'][d['task_type']] += 1
                    dist['attrs'][d['attribute']] += 1
                    dist['qcats'][d.get('question_category', d['attribute'])] += 1
                    dist['tpls'][d['template_id']] += 1
                    dist['ans'][d['answer']] += 1
                    for e in d['entities']: dist['entities'][e] += 1
                    
                    # Chain length
                    if 'attribute' in d['reasoning_relation']:
                        dist['chain_lens'][1] += 1
                    else:
                        try: dist['chain_lens'][len(eval(d['reasoning_relation']))] += 1
                        except: dist['chain_lens'][-1] += 1
                        
                    # Held-out template tracking in test
                    if split == 'test':
                        tids = d['template_id'].split('+')
                        if any(t in UNSEEN_TEMPLATE_IDS for t in tids):
                            test_heldout += 1
                        else:
                            test_familiar += 1

                    # Duplication checks
                    text_sig = d['premise']
                    if text_sig in seen_rendered and seen_rendered[text_sig] != split:
                        all_errors.append(f"[{d['id']}] Text duplicate across {split} and {seen_rendered[text_sig]}")
                    seen_rendered[text_sig] = split
                    
                    rsig = get_reasoning_signature(d)
                    if rsig in seen_reasoning and seen_reasoning[rsig] != split:
                        all_errors.append(f"[{d['id']}] Reasoning-instance duplicate across {split} and {seen_reasoning[rsig]}")
                    seen_reasoning[rsig] = split

            if count != EXPECTED_COUNTS[split]:
                all_errors.append(f"{lang}/{split}: Count {count} != {EXPECTED_COUNTS[split]}")
                
            stats[(lang, split)] = {
                'count': count,
                'dist': dist,
                'tokens': token_lengths,
                'test_fam': test_familiar,
                'test_held': test_heldout,
                'unique_texts': len(seen_rendered),
                'unique_reasoning': len(seen_reasoning)
            }

    # ── Report Generation ──────────────────────────────────────────────────
    lines = ['# Phase 3 Full Dataset Validation Report\n']
    
    if all_errors:
        lines.append(f'## [FAIL] {len(all_errors)} errors found.\n')
        for e in all_errors[:50]: lines.append(f'- {e}')
        if len(all_errors) > 50: lines.append(f'- ... and {len(all_errors)-50} more.')
    else:
        lines.append('## [PASS] No errors found.\n')

    for (lang, split), s in stats.items():
        lines.append(f'### {lang.upper()} / {split.upper()} — {s["count"]} examples')
        
        d = s['dist']
        lines.append(f"  - Tasks: {dict(d['tasks'])}")
        lines.append(f"  - Attributes: {dict(d['attrs'])}")
        lines.append(f"  - Question Categories: {dict(d['qcats'])}")
        lines.append(f"  - Chain Lengths: {dict(d['chain_lens'])}")
        
        # Templates & Entities
        top_tpls = d['tpls'].most_common(5)
        lines.append(f"  - Top Templates: {top_tpls}")
        if top_tpls and top_tpls[0][1] > (s['count'] * 0.15):
            lines.append(f"    *WARNING: Template {top_tpls[0][0]} overrepresented (>15%)*")
            
        top_ents = d['entities'].most_common(5)
        lines.append(f"  - Top Entities: {top_ents}")
        
        # Token Stats
        if s['tokens']:
            t = s['tokens']
            over_limit = sum(1 for x in t if x > MAX_TOKENS)
            lines.append(f"  - Tokens: min={min(t)}, max={max(t)}, mean={statistics.mean(t):.1f}, median={statistics.median(t):.1f}")
            if over_limit > 0: lines.append(f"    *WARNING: {over_limit} examples > {MAX_TOKENS} tokens*")
            
        if split == 'test':
            lines.append(f"  - Test Familiar/Held-out: {s['test_fam']} / {s['test_held']} (target ~50/50)")

        # Unique accumulated tracking (prints at the end of each lang)
        if split == 'test':
            lines.append(f"  - Overall Unique Rendered Texts in {lang}: {s['unique_texts']}")
            lines.append(f"  - Overall Unique Reasoning Instances in {lang}: {s['unique_reasoning']}")
        lines.append('')

    report_path = os.path.join(OUTPUT_DIR, 'dataset_validation_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('\n'.join(lines))
    
    if all_errors: sys.exit(1)
    else: sys.exit(0)

if __name__ == '__main__':
    run_validation()
