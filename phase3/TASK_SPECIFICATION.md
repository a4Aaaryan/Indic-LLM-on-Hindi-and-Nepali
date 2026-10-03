# Phase 3 Task Specification and Template Design

## 1. Task Families

### A. Direct Comparison
- **Structured Representation:** A directed acyclic graph of inequalities between entities. E.g., `[(A, '>', B), (B, '>', C)]`
- **Valid Relations:** `>` (greater), `<` (smaller)
- **Numerical Constraints:** None (purely logical)
- **Question Types:** "Who is the greatest?" / "Who is the smallest?"
- **Answer Type:** Entity Name
- **Computation:** Topological sort or finding the max/min degree node. The entity with in-degree 0 in a purely `>` graph is the greatest.
- **Expected Depth:** 2 to 3 hops (3 to 4 entities).
- **Invalid Structure:** Cycles like `A > B`, `B > C`, `C > A`.

### B. Attribute Comparison
- **Structured Representation:** Map of entities to scalar values. E.g., `{A: 25, B: 30, C: 20}`
- **Valid Relations:** Value assignment.
- **Numerical Constraints:** 
  - Age: integers 10 to 100.
  - Height: integers 100 to 220 (cm).
  - Price: integers 50 to 5000 (rupees).
- **Question Types:** "Who is the oldest/tallest/most expensive?" / "Who is the youngest/shortest/cheapest?"
- **Answer Type:** Entity Name
- **Computation:** `max()` or `min()` on the values.
- **Expected Depth:** 1 hop calculation over 3-5 entities.
- **Invalid Structure:** Duplicate extreme values if the question asks for "the" maximum without a tie-breaker. (We will ensure unique max/min values).

### C. Multi-hop / Transitive Reasoning
- **Structured Representation:** Sequence of relative assignments. E.g., `[(A, '>', B), (B, '=', C), (C, '>', D)]`
- **Valid Relations:** `>`, `<`, `=`
- **Numerical Constraints:** None (logical).
- **Question Types:** "What is the relation between A and D?" (Greater, Smaller, Equal).
- **Answer Type:** Relation word (e.g., 'Greater').
- **Computation:** Path traversal. If all edge weights in path A->D are `>= 0` and at least one is `>`, A > D.
- **Expected Depth:** 3 to 4 hops.
- **Invalid Structure:** Conflicting paths or unlinked entities in the query.

## 2. Hindi Template Bank
**Entities:** राहुल (Rahul), अंजलि (Anjali), विक्रम (Vikram), सुमित (Sumit), नेहा (Neha)
**Attributes:**
- Height (ऊंचाई): लंबा (tall), छोटा (short)
- Age (उम्र): बड़ा (older), छोटा (younger)
- Price (कीमत): महंगा (expensive), सस्ता (cheap)

**Templates for Direct Comparison:**
1. "{E1}, {E2} से {Adj_More} है। {E2}, {E3} से {Adj_More} है।"
   Question: "सबसे {Adj_More} कौन है?"
   Example: "राहुल, अंजलि से लंबा है। अंजलि, विक्रम से लंबी है। सबसे लंबा कौन है?"

**Templates for Attribute Comparison:**
1. "{E1} की उम्र {V1} साल है। {E2} की उम्र {V2} साल है। {E3} की उम्र {V3} साल है।"
   Question: "सबसे बड़ा कौन है?"

## 3. Nepali Template Bank
**Entities:** राम (Ram), सिता (Sita), हरि (Hari), निमा (Nima), आरभ (Aarav)
**Attributes:**
- Height (उचाइ): अग्लो (tall), होचो (short)
- Age (उमेर): जेठो (older), कान्छो (younger)
- Price (मूल्य): महँगो (expensive), सस्तो (cheap)

**Templates for Direct Comparison:**
1. "{E1}, {E2} भन्दा {Adj_More} छ। {E2}, {E3} भन्दा {Adj_More} छ।"
   Question: "सबैभन्दा {Adj_More} को हो?"
   Example: "राम, सिता भन्दा अग्लो छ। सिता, हरि भन्दा अग्लो छ। सबैभन्दा अग्लो को हो?"

**Templates for Attribute Comparison:**
1. "{E1} को उमेर {V1} वर्ष छ। {E2} को उमेर {V2} वर्ष छ। {E3} को उमेर {V3} वर्ष छ।"
   Question: "सबैभन्दा जेठो को हो?"

## 4. Split / Leakage Strategy
- **Entities:** Entities will be strictly partitioned. 
  - Train: Set A of names (e.g., 20 names)
  - Validation: Set B of names (e.g., 5 names)
  - Test: Set C of names (e.g., 5 names)
- **Values:** Test set will use numerical ranges shifted by +10% to test generalization of numbers, or entirely different subsets.
- **Templates:** We will hold out at least one specific formulation (e.g. passive voice or a specific ordering like "{E1} is taller than {E2}, who is taller than {E3}") for the test set.

## 5. Answer Format and EOS Tokenizer Findings
The evaluation answer must be exact match. 
- Format: `[Prompt] उत्तर: [Answer]</s>` or `[Prompt] Answer: [Answer]</s>`. We will use `उत्तर:` (Answer:) in both languages.
- Our tokenizers map the `\n` character to `<unk>` (ID 0). 
- However, the `।` (Purna Viram) is available in both tokenizers (ID 9852 in Hindi, 9869 in Nepali). 
- The `<EOS>` token `</s>` (ID 2) is available.
- We will stop generation at `</s>` or `।` or just calculate exact match up to the end of the expected answer word length.
