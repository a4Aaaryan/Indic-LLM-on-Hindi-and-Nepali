"""
Phase 3 — Synthetic Reasoning Dataset Generator
================================================
Deterministic, attribute-consistent generation pipeline.

Core invariants enforced throughout:
    attribute(premise_templates)  == attribute(question)  == attribute(answer)
    question_category(template)   == question_category(question text chosen)

Every template carries explicit 'attribute' AND 'question_category' tags.
For non-quantity attributes, question_category == attribute.
For quantity, question_category distinguishes:
    'possession_count'  – people owning books / items
    'substance_amount'  – measurable substances (water, salt, oil …)
    'object_count'      – countable discrete objects
"""

import json
import os
import random
import sys
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from reasoning_solvers import ReasoningSolver

# ---------------------------------------------------------------------------
# 1.  TEMPLATE DEFINITIONS
#     Every entry has:
#       text      – raw template string with {E1}, {E2}, {E3}, {V1} slots
#       task      – 'direct' | 'attribute' | 'transitive'
#       attribute – 'height' | 'age' | 'price' | 'quantity' | 'generic_order'
#       rel       – canonical operator:  '>' | '<' | '=' | 'mixed'
#       entities  – 'people' | 'objects' | 'substances' | 'people_or_objects'
#       multi     – (optional) True when template encodes >1 relation
# ---------------------------------------------------------------------------

HINDI_TEMPLATES = {
    # ── HEIGHT ──────────────────────────────────────────────────────────────
    'H1':  {'text': '{E1} की लंबाई {E2} से ज़्यादा है।',
            'task': 'direct', 'attribute': 'height', 'rel': '>', 'entities': 'people_or_objects'},
    'H2':  {'text': '{E2} का कद {E1} से कम है।',
            'task': 'direct', 'attribute': 'height', 'rel': '<', 'entities': 'people'},
    'H3':  {'text': '{E2} और {E3} दोनों, {E1} से लंबे हैं।',
            'task': 'direct', 'attribute': 'height', 'rel': 'mixed', 'entities': 'people_or_objects', 'multi': True},
    'H4':  {'text': '{E1}, {E2} के मुकाबले कद में छोटा है।',
            'task': 'direct', 'attribute': 'height', 'rel': '<', 'entities': 'people'},
    'H5':  {'text': '{E1} की लंबाई {E2} के मुकाबले कम है।',
            'task': 'direct', 'attribute': 'height', 'rel': '<', 'entities': 'people_or_objects'},
    'H6':  {'text': '{E1}, {E2} से ज़्यादा लंबा है।',
            'task': 'direct', 'attribute': 'height', 'rel': '>', 'entities': 'people_or_objects'},
    'H7':  {'text': '{E1}, {E2} से लंबा है, पर {E3} दोनों से छोटा है।',
            'task': 'direct', 'attribute': 'height', 'rel': 'mixed', 'entities': 'people_or_objects', 'multi': True},
    'H9':  {'text': '{E1} की लंबाई {E2} से ज़्यादा है और {E3} से कम है।',
            'task': 'direct', 'attribute': 'height', 'rel': 'mixed', 'entities': 'people_or_objects', 'multi': True},
    'H10': {'text': '{E1} की तुलना में {E2} अधिक लंबा है।',
            'task': 'direct', 'attribute': 'height', 'rel': '<', 'entities': 'people_or_objects'},
    'H11': {'text': '{E1} का कद {E2} से ऊंचा है।',
            'task': 'direct', 'attribute': 'height', 'rel': '>', 'entities': 'people'},
    'H12': {'text': '{E1} और {E2} में से {E1} लंबा है।',
            'task': 'direct', 'attribute': 'height', 'rel': '>', 'entities': 'people_or_objects'},
    'H14': {'text': '{E1} की ऊंचाई {E2} और {E3} दोनों से अधिक है।',
            'task': 'direct', 'attribute': 'height', 'rel': 'mixed', 'entities': 'people_or_objects', 'multi': True},
    # ── AGE ─────────────────────────────────────────────────────────────────
    'H16': {'text': '{E1} की उम्र {V1} साल है।',
            'task': 'attribute', 'attribute': 'age', 'rel': 'numeric', 'entities': 'people'},
    'H17': {'text': '{E1}, {E2} से उम्र में बड़ा है।',
            'task': 'direct', 'attribute': 'age', 'rel': '>', 'entities': 'people'},
    'H18': {'text': '{E1} की उम्र {E2} से कम है।',
            'task': 'direct', 'attribute': 'age', 'rel': '<', 'entities': 'people'},
    'H19': {'text': '{E1}, {E2} से उम्र में छोटा है, लेकिन {E3} उससे भी छोटा है।',
            'task': 'direct', 'attribute': 'age', 'rel': 'mixed', 'entities': 'people', 'multi': True},
    'H20': {'text': '{E1} और {E2} की उम्र बराबर है।',
            'task': 'transitive', 'attribute': 'age', 'rel': '=', 'entities': 'people'},
    'H22': {'text': '{E1} की उम्र {E2} से अधिक लेकिन {E3} से कम है।',
            'task': 'direct', 'attribute': 'age', 'rel': 'mixed', 'entities': 'people', 'multi': True},
    'H23': {'text': '{E1} और {E2} की उम्र समान है, जबकि {E3} दोनों से बड़ा है।',
            'task': 'transitive', 'attribute': 'age', 'rel': 'mixed', 'entities': 'people', 'multi': True},
    # ── PRICE ───────────────────────────────────────────────────────────────
    'H24': {'text': '{E1} की कीमत {V1} रुपये है।',
            'task': 'attribute', 'attribute': 'price', 'rel': 'numeric', 'entities': 'objects'},
    'H25': {'text': '{E1}, {E2} से महंगा है।',
            'task': 'direct', 'attribute': 'price', 'rel': '>', 'entities': 'objects'},
    'H26': {'text': '{E1} की कीमत {E2} से कम है।',
            'task': 'direct', 'attribute': 'price', 'rel': '<', 'entities': 'objects'},
    'H27': {'text': '{E1} और {E2} में से {E1} अधिक महंगा है।',
            'task': 'direct', 'attribute': 'price', 'rel': '>', 'entities': 'objects'},
    'H29': {'text': '{E1} की कीमत {E2} और {E3} दोनों से अधिक है।',
            'task': 'direct', 'attribute': 'price', 'rel': 'mixed', 'entities': 'objects', 'multi': True},
    'H30': {'text': '{E1} की कीमत {E2} से अधिक लेकिन {E3} से कम है।',
            'task': 'direct', 'attribute': 'price', 'rel': 'mixed', 'entities': 'objects', 'multi': True},
    'H31': {'text': '{E1} और {E2} की कीमत समान है, जबकि {E3} की कीमत दोनों से अधिक है।',
            'task': 'transitive', 'attribute': 'price', 'rel': 'mixed', 'entities': 'objects', 'multi': True},
    # ── QUANTITY ────────────────────────────────────────────────────────────
    # People + book/item possessions
    'H32': {'text': '{E1} के पास {V1} किताबें हैं।',
            'task': 'attribute', 'attribute': 'quantity', 'question_category': 'possession_count',
            'rel': 'numeric', 'entities': 'people'},
    'H33': {'text': '{E1} के पास {E2} से अधिक किताबें हैं।',
            'task': 'direct', 'attribute': 'quantity', 'question_category': 'possession_count',
            'rel': '>', 'entities': 'people'},
    'H34': {'text': '{E1} के पास {E2} से कम सामान है।',
            'task': 'direct', 'attribute': 'quantity', 'question_category': 'possession_count',
            'rel': '<', 'entities': 'people'},
    # Substances + measurable amount
    'H35': {'text': '{E1} में {E2} की तुलना में अधिक मात्रा है।',
            'task': 'direct', 'attribute': 'quantity', 'question_category': 'substance_amount',
            'rel': '>', 'entities': 'substances'},
    'H38': {'text': '{E1} की मात्रा {E2} से अधिक लेकिन {E3} से कम है।',
            'task': 'direct', 'attribute': 'quantity', 'question_category': 'substance_amount',
            'rel': 'mixed', 'entities': 'substances', 'multi': True},
    # Countable objects + object-count
    'H36': {'text': '{E1} की संख्या {E2} से अधिक है।',
            'task': 'direct', 'attribute': 'quantity', 'question_category': 'object_count',
            'rel': '>', 'entities': 'objects'},
    'H37': {'text': '{E1} और {E2} की संख्या बराबर है।',
            'task': 'transitive', 'attribute': 'quantity', 'question_category': 'object_count',
            'rel': '=', 'entities': 'objects'},
    # ── GENERIC ORDER ────────────────────────────────────────────────────────
    'H40': {'text': '{E1}, {E2} से बड़ा है और {E2}, {E3} से बड़ा है।',
            'task': 'transitive', 'attribute': 'generic_order', 'rel': 'mixed', 'entities': 'people_or_objects', 'multi': True},
    'H41': {'text': '{E1}, {E2} से छोटा है, जबकि {E3}, {E2} से बड़ा है।',
            'task': 'transitive', 'attribute': 'generic_order', 'rel': 'mixed', 'entities': 'people_or_objects', 'multi': True},
    'H42': {'text': '{E1}, {E2} से आगे है और {E3}, {E1} से आगे है।',
            'task': 'transitive', 'attribute': 'generic_order', 'rel': 'mixed', 'entities': 'people_or_objects', 'multi': True},
    'H43': {'text': '{E1}, {E2} से ऊपर है और {E3}, {E2} से नीचे है।',
            'task': 'transitive', 'attribute': 'generic_order', 'rel': 'mixed', 'entities': 'objects', 'multi': True},
    'H44': {'text': '{E1} और {E2} बराबर हैं, लेकिन {E3} दोनों से अधिक है।',
            'task': 'transitive', 'attribute': 'generic_order', 'rel': 'mixed', 'entities': 'people_or_objects', 'multi': True},
    'H45': {'text': '{E1}, {E2} से अधिक है और {E3}, {E1} से कम है।',
            'task': 'transitive', 'attribute': 'generic_order', 'rel': 'mixed', 'entities': 'people_or_objects', 'multi': True},
}

NEPALI_TEMPLATES = {
    # ── HEIGHT ──────────────────────────────────────────────────────────────
    'N1':  {'text': '{E1} को लम्बाइ {E2} को भन्दा बढी छ।',
            'task': 'direct', 'attribute': 'height', 'rel': '>', 'entities': 'people_or_objects'},
    'N2':  {'text': '{E2} को कद {E1} को भन्दा कम छ।',
            'task': 'direct', 'attribute': 'height', 'rel': '<', 'entities': 'people'},
    'N3':  {'text': '{E2} र {E3} दुवै {E1} भन्दा अग्ला छन्।',
            'task': 'direct', 'attribute': 'height', 'rel': 'mixed', 'entities': 'people_or_objects', 'multi': True},
    'N4':  {'text': '{E1}, {E2} को तुलनामा होचो छ।',
            'task': 'direct', 'attribute': 'height', 'rel': '<', 'entities': 'people'},
    'N5':  {'text': '{E1} को लम्बाइ {E2} को तुलनामा कम छ।',
            'task': 'direct', 'attribute': 'height', 'rel': '<', 'entities': 'people_or_objects'},
    'N6':  {'text': '{E1}, {E2} भन्दा अग्लो छ।',
            'task': 'direct', 'attribute': 'height', 'rel': '>', 'entities': 'people_or_objects'},
    'N7':  {'text': '{E1}, {E2} भन्दा अग्लो छ, तर {E3} दुवैभन्दा होचो छ।',
            'task': 'direct', 'attribute': 'height', 'rel': 'mixed', 'entities': 'people_or_objects', 'multi': True},
    'N9':  {'text': '{E1} को लम्बाइ {E2} को भन्दा बढी छ र {E3} को भन्दा कम छ।',
            'task': 'direct', 'attribute': 'height', 'rel': 'mixed', 'entities': 'people_or_objects', 'multi': True},
    'N10': {'text': '{E1} को तुलनामा {E2} अग्लो छ।',
            'task': 'direct', 'attribute': 'height', 'rel': '<', 'entities': 'people_or_objects'},
    'N11': {'text': '{E1} को कद {E2} को भन्दा अग्लो छ।',
            'task': 'direct', 'attribute': 'height', 'rel': '>', 'entities': 'people'},
    'N12': {'text': '{E1} र {E2} मध्ये {E1} अग्लो छ।',
            'task': 'direct', 'attribute': 'height', 'rel': '>', 'entities': 'people_or_objects'},
    'N14': {'text': '{E1} को उचाइ {E2} र {E3} दुवैको भन्दा बढी छ।',
            'task': 'direct', 'attribute': 'height', 'rel': 'mixed', 'entities': 'people_or_objects', 'multi': True},
    # ── AGE ─────────────────────────────────────────────────────────────────
    'N16': {'text': '{E1} को उमेर {V1} वर्ष हो।',
            'task': 'attribute', 'attribute': 'age', 'rel': 'numeric', 'entities': 'people'},
    'N17': {'text': '{E1}, {E2} भन्दा उमेरमा जेठो छ।',
            'task': 'direct', 'attribute': 'age', 'rel': '>', 'entities': 'people'},
    'N18': {'text': '{E1} को उमेर {E2} को भन्दा कम छ।',
            'task': 'direct', 'attribute': 'age', 'rel': '<', 'entities': 'people'},
    'N19': {'text': '{E1}, {E2} भन्दा उमेरमा कान्छो छ, तर {E3} त्योभन्दा पनि कान्छो छ।',
            'task': 'direct', 'attribute': 'age', 'rel': 'mixed', 'entities': 'people', 'multi': True},
    'N20': {'text': '{E1} र {E2} को उमेर बराबर छ।',
            'task': 'transitive', 'attribute': 'age', 'rel': '=', 'entities': 'people'},
    'N22': {'text': '{E1} को उमेर {E2} को भन्दा बढी तर {E3} को भन्दा कम छ।',
            'task': 'direct', 'attribute': 'age', 'rel': 'mixed', 'entities': 'people', 'multi': True},
    'N23': {'text': '{E1} र {E2} को उमेर समान छ, जबकि {E3} दुवैभन्दा जेठो छ।',
            'task': 'transitive', 'attribute': 'age', 'rel': 'mixed', 'entities': 'people', 'multi': True},
    # ── PRICE ───────────────────────────────────────────────────────────────
    'N24': {'text': '{E1} को मूल्य {V1} रुपैयाँ छ।',
            'task': 'attribute', 'attribute': 'price', 'rel': 'numeric', 'entities': 'objects'},
    'N25': {'text': '{E1}, {E2} भन्दा महँगो छ।',
            'task': 'direct', 'attribute': 'price', 'rel': '>', 'entities': 'objects'},
    'N26': {'text': '{E1} को मूल्य {E2} को भन्दा कम छ।',
            'task': 'direct', 'attribute': 'price', 'rel': '<', 'entities': 'objects'},
    'N27': {'text': '{E1} र {E2} मध्ये {E1} बढी महँगो छ।',
            'task': 'direct', 'attribute': 'price', 'rel': '>', 'entities': 'objects'},
    'N29': {'text': '{E1} को मूल्य {E2} र {E3} दुवैको भन्दा बढी छ।',
            'task': 'direct', 'attribute': 'price', 'rel': 'mixed', 'entities': 'objects', 'multi': True},
    'N30': {'text': '{E1} को मूल्य {E2} को भन्दा बढी तर {E3} को भन्दा कम छ।',
            'task': 'direct', 'attribute': 'price', 'rel': 'mixed', 'entities': 'objects', 'multi': True},
    'N31': {'text': '{E1} र {E2} को मूल्य समान छ, जबकि {E3} को मूल्य दुवैको भन्दा बढी छ।',
            'task': 'transitive', 'attribute': 'price', 'rel': 'mixed', 'entities': 'objects', 'multi': True},
    # ── QUANTITY ────────────────────────────────────────────────────────────
    # People + book/item possessions
    'N32': {'text': '{E1} सँग {V1} वटा किताब छन्।',
            'task': 'attribute', 'attribute': 'quantity', 'question_category': 'possession_count',
            'rel': 'numeric', 'entities': 'people'},
    'N33': {'text': '{E1} सँग {E2} को भन्दा धेरै किताब छन्।',
            'task': 'direct', 'attribute': 'quantity', 'question_category': 'possession_count',
            'rel': '>', 'entities': 'people'},
    'N34': {'text': '{E1} सँग {E2} को भन्दा कम सामान छ।',
            'task': 'direct', 'attribute': 'quantity', 'question_category': 'possession_count',
            'rel': '<', 'entities': 'people'},
    # Substances + measurable amount
    'N35': {'text': '{E1} मा {E2} को तुलनामा बढी मात्रा छ।',
            'task': 'direct', 'attribute': 'quantity', 'question_category': 'substance_amount',
            'rel': '>', 'entities': 'substances'},
    'N38': {'text': '{E1} को मात्रा {E2} को भन्दा बढी तर {E3} को भन्दा कम छ।',
            'task': 'direct', 'attribute': 'quantity', 'question_category': 'substance_amount',
            'rel': 'mixed', 'entities': 'substances', 'multi': True},
    # Countable objects + object-count
    'N36': {'text': '{E1} को संख्या {E2} को भन्दा बढी छ।',
            'task': 'direct', 'attribute': 'quantity', 'question_category': 'object_count',
            'rel': '>', 'entities': 'objects'},
    'N37': {'text': '{E1} र {E2} को संख्या बराबर छ।',
            'task': 'transitive', 'attribute': 'quantity', 'question_category': 'object_count',
            'rel': '=', 'entities': 'objects'},
    # ── GENERIC ORDER ────────────────────────────────────────────────────────
    'N40': {'text': '{E1}, {E2} भन्दा ठूलो छ र {E2}, {E3} भन्दा ठूलो छ।',
            'task': 'transitive', 'attribute': 'generic_order', 'rel': 'mixed', 'entities': 'people_or_objects', 'multi': True},
    'N41': {'text': '{E1}, {E2} भन्दा सानो छ, जबकि {E3}, {E2} भन्दा ठूलो छ।',
            'task': 'transitive', 'attribute': 'generic_order', 'rel': 'mixed', 'entities': 'people_or_objects', 'multi': True},
    'N42': {'text': '{E1}, {E2} भन्दा अगाडि छ र {E3}, {E1} भन्दा अगाडि छ।',
            'task': 'transitive', 'attribute': 'generic_order', 'rel': 'mixed', 'entities': 'people_or_objects', 'multi': True},
    'N43': {'text': '{E1}, {E2} भन्दा माथि छ र {E3}, {E2} भन्दा तल छ।',
            'task': 'transitive', 'attribute': 'generic_order', 'rel': 'mixed', 'entities': 'objects', 'multi': True},
    'N44': {'text': '{E1} र {E2} बराबर छन्, तर {E3} दुवैभन्दा बढी छ।',
            'task': 'transitive', 'attribute': 'generic_order', 'rel': 'mixed', 'entities': 'people_or_objects', 'multi': True},
    'N45': {'text': '{E1}, {E2} भन्दा बढी छ र {E3}, {E1} भन्दा कम छ।',
            'task': 'transitive', 'attribute': 'generic_order', 'rel': 'mixed', 'entities': 'people_or_objects', 'multi': True},
}

# ---------------------------------------------------------------------------
# 2.  QUESTION / ANSWER TEXT TABLES   (keyed by attribute × language × q_type)
#     This completely decouples question rendering from template text.
# ---------------------------------------------------------------------------

# For non-quantity attributes, question_category == attribute.
# For quantity, question_category is one of:
#   possession_count | substance_amount | object_count
QUESTION_TEXT = {
    'hindi': {
        'height':           {'max': 'सबसे लंबा कौन है?',
                             'min': 'सबसे छोटा कौन है?'},
        'age':              {'max': 'उम्र में सबसे बड़ा कौन है?',
                             'min': 'उम्र में सबसे छोटा कौन है?'},
        'price':            {'max': 'सबसे महंगा क्या है?',
                             'min': 'सबसे सस्ता क्या है?'},
        'generic_order':    {'max': 'सबसे आगे / ऊपर कौन है?',
                             'min': 'सबसे पीछे / नीचे कौन है?'},
        # quantity sub-categories
        'possession_count': {'max': 'किसके पास सबसे अधिक किताबें हैं?',
                             'min': 'किसके पास सबसे कम किताबें हैं?'},
        'substance_amount': {'max': 'किसमें सबसे अधिक मात्रा है?',
                             'min': 'किसमें सबसे कम मात्रा है?'},
        'object_count':     {'max': 'किसकी संख्या सबसे अधिक है?',
                             'min': 'किसकी संख्या सबसे कम है?'},
    },
    'nepali': {
        'height':           {'max': 'सबैभन्दा अग्लो को हो?',
                             'min': 'सबैभन्दा होचो को हो?'},
        'age':              {'max': 'उमेरमा सबैभन्दा जेठो को हो?',
                             'min': 'उमेरमा सबैभन्दा कान्छो को हो?'},
        'price':            {'max': 'सबैभन्दा महँगो के हो?',
                             'min': 'सबैभन्दा सस्तो के हो?'},
        'generic_order':    {'max': 'सबैभन्दा माथि / अगाडि को हो?',
                             'min': 'सबैभन्दा तल / पछाडि को हो?'},
        # quantity sub-categories
        'possession_count': {'max': 'कोसँग सबैभन्दा धेरै किताबहरू छन्?',
                             'min': 'कोसँग सबैभन्दा कम किताबहरू छन्?'},
        'substance_amount': {'max': 'कुनमा सबैभन्दा बढी मात्रा छ?',
                             'min': 'कुनमा सबैभन्दा कम मात्रा छ?'},
        'object_count':     {'max': 'कसको संख्या सबैभन्दा बढी छ?',
                             'min': 'कसको संख्या सबैभन्दा कम छ?'},
    },
}

# For transitive examples: relation words keyed by attribute
RELATION_ANSWER_TEXT = {
    'hindi': {
        'height':        {'>': 'लंबा', '<': 'छोटा', '=': 'बराबर'},
        'age':           {'>': 'बड़ा',  '<': 'छोटा', '=': 'बराबर'},
        'price':         {'>': 'महंगा', '<': 'सस्ता', '=': 'समान'},
        'quantity':      {'>': 'अधिक', '<': 'कम',    '=': 'बराबर'},
        'generic_order': {'>': 'बड़ा',  '<': 'छोटा', '=': 'बराबर'},
    },
    'nepali': {
        'height':        {'>': 'अग्लो', '<': 'होचो',  '=': 'बराबर'},
        'age':           {'>': 'जेठो',  '<': 'कान्छो','=': 'बराबर'},
        'price':         {'>': 'महँगो', '<': 'सस्तो', '=': 'समान'},
        'quantity':      {'>': 'धेरै',  '<': 'कम',    '=': 'बराबर'},
        'generic_order': {'>': 'ठूलो',  '<': 'सानो',  '=': 'बराबर'},
    },
}

TRANSITIVE_Q_TEXT = {
    'hindi':  '{E1} और {E2} में क्या संबंध है?',
    'nepali': '{E1} र {E2} बीच के सम्बन्ध छ?',
}

# ---------------------------------------------------------------------------
# 3.  ENTITY POOLS  (strict disjoint sets per split)
# ---------------------------------------------------------------------------

ENTITIES = {
    'hindi': {
        'people': {
            'train': ['राहुल', 'अमित', 'विक्रम', 'नेहा', 'सुमित', 'अंजलि', 'पूजा', 'रोहित',
                      'रवि', 'सुनीता', 'कपिल', 'गीता', 'मोहन', 'सोनिया', 'प्रदीप',
                      'नीता', 'सुरेश', 'किरण', 'अशोक', 'ममता', 'संदीप', 'राधा'],
            'val':   ['राजू', 'अनीता', 'संजय', 'सीता', 'विजय'],
            'test':  ['मनोज', 'रचना', 'दीपक', 'आशा', 'दिनेश'],
        },
        'objects': {
            'train': ['किताब', 'कलम', 'बैग', 'कुर्सी', 'मेज', 'मोबाइल', 'कंप्यूटर',
                      'घड़ी', 'बोतल', 'जूता', 'गाड़ी', 'साइकिल', 'पंखा', 'टीवी', 'अलमारी'],
            'val':   ['चश्मा', 'पेन', 'कॉपी', 'दर्पण', 'चाबी'],
            'test':  ['लैपटॉप', 'रेडियो', 'कैमरा', 'गिलास', 'चम्मच'],
        },
        'substances': {
            'train': ['पानी', 'दूध', 'तेल', 'चीनी', 'नमक', 'चावल', 'दाल', 'आटा'],
            'val':   ['शहद', 'घी', 'चाय'],
            'test':  ['जूस', 'कॉफी', 'रेत'],
        },
    },
    'nepali': {
        'people': {
            'train': ['राम', 'हरि', 'सिता', 'गीता', 'श्याम', 'गोपाल', 'निमा', 'आरभ',
                      'कृष्ण', 'सरस्वती', 'रमेश', 'सुनिता', 'प्रकाश', 'पार्वती', 'सुभाष',
                      'मञ्जु', 'नारायण', 'कमला', 'प्रमोद', 'विमला', 'राजेश', 'लक्ष्मी'],
            'val':   ['नवीन', 'सृजना', 'दिपेश', 'अस्मिता', 'सन्तोष'],
            'test':  ['विकास', 'संगीता', 'सुरज', 'पुष्पा', 'विनोद'],
        },
        'objects': {
            'train': ['किताब', 'कलम', 'झोला', 'कुर्सी', 'टेबुल', 'मोबाइल', 'कम्प्युटर',
                      'घडी', 'बोतल', 'जुत्ता', 'गाडी', 'साइकल', 'पंखा', 'टिभी', 'दराज'],
            'val':   ['चस्मा', 'कापी', 'ऐना', 'साँचो', 'टोपी'],
            'test':  ['ल्यापटप', 'रेडियो', 'क्यामेरा', 'गिलास', 'चम्चा'],
        },
        'substances': {
            'train': ['पानी', 'दूध', 'तेल', 'चिनी', 'नुन', 'चामल', 'दाल', 'पिठो'],
            'val':   ['मह', 'घिउ', 'चिया'],
            'test':  ['जुस', 'कफी', 'बालुवा'],
        },
    },
}

# ---------------------------------------------------------------------------
# 4.  MULTI-TEMPLATE RELATION MAPS
#     Specifies the exact relation list for every multi-fact template.
# ---------------------------------------------------------------------------

MULTI_RELS = {
    'H3':  lambda e: [(e[1], '>', e[0]), (e[2], '>', e[0])],
    'H7':  lambda e: [(e[0], '>', e[1]), (e[2], '<', e[0]), (e[2], '<', e[1])],
    'H9':  lambda e: [(e[0], '>', e[1]), (e[0], '<', e[2])],
    'H14': lambda e: [(e[0], '>', e[1]), (e[0], '>', e[2])],
    'H19': lambda e: [(e[0], '<', e[1]), (e[2], '<', e[0])],
    'H22': lambda e: [(e[0], '>', e[1]), (e[0], '<', e[2])],
    'H23': lambda e: [(e[0], '=', e[1]), (e[2], '>', e[0]), (e[2], '>', e[1])],
    'H29': lambda e: [(e[0], '>', e[1]), (e[0], '>', e[2])],
    'H30': lambda e: [(e[0], '>', e[1]), (e[0], '<', e[2])],
    'H31': lambda e: [(e[0], '=', e[1]), (e[2], '>', e[0]), (e[2], '>', e[1])],
    'H38': lambda e: [(e[0], '>', e[1]), (e[0], '<', e[2])],
    'H40': lambda e: [(e[0], '>', e[1]), (e[1], '>', e[2])],
    'H41': lambda e: [(e[0], '<', e[1]), (e[2], '>', e[1])],
    'H42': lambda e: [(e[0], '>', e[1]), (e[2], '>', e[0])],
    'H43': lambda e: [(e[0], '>', e[1]), (e[2], '<', e[1])],
    'H44': lambda e: [(e[0], '=', e[1]), (e[2], '>', e[0])],
    'H45': lambda e: [(e[0], '>', e[1]), (e[2], '<', e[0])],
}
# Nepali versions share the same relation logic
for hid, nid in [('H3','N3'), ('H7','N7'), ('H9','N9'), ('H14','N14'), ('H19','N19'),
                 ('H22','N22'), ('H23','N23'), ('H29','N29'), ('H30','N30'), ('H31','N31'),
                 ('H38','N38'), ('H40','N40'), ('H41','N41'), ('H42','N42'),
                 ('H43','N43'), ('H44','N44'), ('H45','N45')]:
    MULTI_RELS[nid] = MULTI_RELS[hid]

# ---------------------------------------------------------------------------
# 5.  HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def get_entities(lang, cat, split, count):
    """Return `count` distinct entities from the correct disjoint pool."""
    resolved_cat = random.choice(['people', 'objects']) if cat == 'people_or_objects' else cat
    pool = ENTITIES[lang][resolved_cat][split]
    if len(pool) < count:
        raise ValueError(f"Not enough entities: {lang}-{resolved_cat}-{split} (need {count}, have {len(pool)})")
    return random.sample(pool, count), resolved_cat


def fill_template(text, entities, value=None):
    """Substitute {E1}…{En} and {V1} into a template text."""
    for i, e in enumerate(entities):
        text = text.replace(f'{{E{i+1}}}', e)
    if value is not None:
        text = text.replace('{V1}', str(value))
    assert '{' not in text, f"Unresolved placeholder in: {text}"
    return text


def slot_count(text):
    """Count {E…} slots in template text."""
    return sum(1 for i in range(1, 10) if f'{{E{i}}}' in text)

# ---------------------------------------------------------------------------
# 6.  DATASET GENERATOR
# ---------------------------------------------------------------------------

# Templates only available in TEST (structural novelty)
UNSEEN_TEMPLATE_IDS = {'H3', 'H7', 'H9', 'H22', 'H31', 'H40',
                       'N3', 'N7', 'N9', 'N22', 'N31', 'N40'}


class DatasetGenerator:
    def __init__(self, lang, split):
        self.lang = lang
        self.split = split
        self.all_templates = HINDI_TEMPLATES if lang == 'hindi' else NEPALI_TEMPLATES
        self.solver = ReasoningSolver()

        # Group template IDs by attribute for same-attribute sampling
        self.by_attr = {}
        for tid, tpl in self.all_templates.items():
            attr = tpl['attribute']
            self.by_attr.setdefault(attr, []).append(tid)

        # Split-aware seen / unseen sets
        self.seen_by_attr   = {a: [t for t in ids if t not in UNSEEN_TEMPLATE_IDS]
                                for a, ids in self.by_attr.items()}
        self.unseen_by_attr = {a: [t for t in ids if t in UNSEEN_TEMPLATE_IDS]
                                for a, ids in self.by_attr.items()}

    # ── entry point ──────────────────────────────────────────────────────────

    def generate(self, count):
        results = []
        attempts = 0
        while len(results) < count and attempts < 20_000:
            attempts += 1
            item = self._try_one()
            if item is not None:
                item['id']        = f"{self.lang}_{self.split}_{len(results)}"
                item['language']  = self.lang
                item['split']     = self.split
                item['unk_count'] = 0
                results.append(item)
        return results

    def _try_one(self):
        # Pick a random attribute bucket
        attr = random.choice(list(self.by_attr.keys()))

        # Pick template pool based on split
        if self.split == 'test' and random.random() < 0.5:
            pool = self.unseen_by_attr.get(attr, [])
        else:
            pool = self.seen_by_attr.get(attr, [])
        if not pool:
            return None

        tid  = random.choice(pool)
        tpl  = self.all_templates[tid]
        task = tpl['task']

        try:
            if task == 'attribute':
                return self._gen_attribute(tid, tpl, attr)
            elif task == 'direct':
                return self._gen_direct(tid, tpl, attr)
            elif task == 'transitive':
                return self._gen_transitive(tid, tpl, attr)
        except (ValueError, AssertionError):
            return None

    # ── attribute task ───────────────────────────────────────────────────────

    def _gen_attribute(self, tid, tpl, attr):
        entities, resolved_cat = get_entities(self.lang, tpl['entities'], self.split, 3)
        values = random.sample(range(10, 100), 3)

        sentences = []
        attr_map  = {}
        for ent, val in zip(entities, values):
            sentences.append(fill_template(tpl['text'], [ent], value=val))
            attr_map[ent] = val

        q_type = random.choice(['max', 'min'])
        answer = self.solver.solve_attribute_comparison(attr_map, q_type)
        if answer is None:
            return None

        q_cat = tpl.get('question_category', attr)
        return {
            'task_type':        task_from_tpl(tpl),
            'attribute':        attr,
            'question_category': q_cat,
            'template_id':      tid,
            'entity_category':  resolved_cat,
            'premise':          ' '.join(sentences),
            'question':         QUESTION_TEXT[self.lang][q_cat][q_type],
            'answer':           answer,
            'entities':         entities,
            'values':           attr_map,
            'reasoning_relation': f'attribute:{q_type}',
        }

    # ── direct task ──────────────────────────────────────────────────────────

    def _gen_direct(self, tid, tpl, attr):
        if tpl.get('multi'):
            return self._gen_direct_multi(tid, tpl, attr)
        else:
            return self._gen_direct_chain(tid, tpl, attr)

    def _gen_direct_multi(self, tid, tpl, attr):
        n_slots = slot_count(tpl['text'])
        entities, resolved_cat = get_entities(self.lang, tpl['entities'], self.split, n_slots)
        text = fill_template(tpl['text'], entities)

        if tid not in MULTI_RELS:
            return None
        rels = MULTI_RELS[tid](entities)

        q_type = random.choice(['max', 'min'])
        answer = self.solver.solve_direct_comparison(rels, q_type)
        if answer is None:
            return None

        q_cat = tpl.get('question_category', attr)
        return {
            'task_type':        task_from_tpl(tpl),
            'attribute':        attr,
            'question_category': q_cat,
            'template_id':      tid,
            'entity_category':  resolved_cat,
            'premise':          text,
            'question':         QUESTION_TEXT[self.lang][q_cat][q_type],
            'answer':           answer,
            'entities':         entities,
            'values':           {},
            'reasoning_relation': str(rels),
        }

    def _gen_direct_chain(self, tid, tpl, attr):
        # Chain two attribute-compatible, SAME-QUESTION-CATEGORY simple templates
        q_cat = tpl.get('question_category', attr)
        same_attr_simple = [k for k, v in self.all_templates.items()
                             if v['attribute'] == attr
                             and v.get('question_category', attr) == q_cat
                             and v['task'] == 'direct'
                             and not v.get('multi')
                             and v['entities'] == tpl['entities']]
        if not same_attr_simple:
            return None
        tid2 = random.choice(same_attr_simple)
        tpl2 = self.all_templates[tid2]

        entities, resolved_cat = get_entities(self.lang, tpl['entities'], self.split, 3)

        s1 = fill_template(tpl['text'],  [entities[0], entities[1]])
        s2 = fill_template(tpl2['text'], [entities[1], entities[2]])

        rel1 = (entities[0], tpl['rel'],  entities[1])
        rel2 = (entities[1], tpl2['rel'], entities[2])

        q_type = random.choice(['max', 'min'])
        answer = self.solver.solve_direct_comparison([rel1, rel2], q_type)
        if answer is None:
            return None

        return {
            'task_type':        task_from_tpl(tpl),
            'attribute':        attr,
            'question_category': q_cat,
            'template_id':      f"{tid}+{tid2}",
            'entity_category':  resolved_cat,
            'premise':          f"{s1} {s2}",
            'question':         QUESTION_TEXT[self.lang][q_cat][q_type],
            'answer':           answer,
            'entities':         entities,
            'values':           {},
            'reasoning_relation': str([rel1, rel2]),
        }

    # ── transitive task ──────────────────────────────────────────────────────

    def _gen_transitive(self, tid, tpl, attr):
        if not tpl.get('multi'):
            return None  # single-equality templates not used standalone
        n_slots = slot_count(tpl['text'])
        entities, resolved_cat = get_entities(self.lang, tpl['entities'], self.split, n_slots)
        text = fill_template(tpl['text'], entities)

        if tid not in MULTI_RELS:
            return None
        rels = MULTI_RELS[tid](entities)

        q_e1, q_e2 = entities[0], entities[-1]
        ans_sym = self.solver.solve_transitive(rels, q_e1, q_e2)
        if ans_sym == '?':
            return None

        ans_word = RELATION_ANSWER_TEXT[self.lang][attr].get(ans_sym)
        if ans_word is None:
            return None

        q_cat = tpl.get('question_category', attr)
        q_text = TRANSITIVE_Q_TEXT[self.lang].replace('{E1}', q_e1).replace('{E2}', q_e2)

        return {
            'task_type':        task_from_tpl(tpl),
            'attribute':        attr,
            'question_category': q_cat,
            'template_id':      tid,
            'entity_category':  resolved_cat,
            'premise':          text,
            'question':         q_text,
            'answer':           ans_word,
            'entities':         entities,
            'values':           {},
            'reasoning_relation': str(rels),
        }


def task_from_tpl(tpl):
    return tpl['task']


# ---------------------------------------------------------------------------
# 7.  MAIN
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    random.seed(42)
    OUTPUT_DIR = 'results/phase3'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    summary = {}
    for lang in ['hindi', 'nepali']:
        summary[lang] = {}
        for split, count in [('train', 9600), ('val', 1200), ('test', 1200)]:
            gen  = DatasetGenerator(lang, split)
            data = gen.generate(count)
            out  = f'{OUTPUT_DIR}/{lang}_{split}.jsonl'
            with open(out, 'w', encoding='utf-8') as f:
                for d in data:
                    f.write(json.dumps(d, ensure_ascii=False) + '\n')
            summary[lang][split] = len(data)
            print(f"  {lang}/{split}: {len(data)} examples written.")

    with open(f'{OUTPUT_DIR}/dataset_generation_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print("Generation complete.")
