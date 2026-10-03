# Phase 2 Architecture

## 1. Model Overview
We are training two completely independent decoder-only language models (Model H for Hindi, Model L for Nepali). Both models use the exact same architecture to ensure a controlled comparison. There is no transfer learning, no parameter sharing, and no shared checkpoints between the two languages. 

## 2. Exact Architecture Table

| Hyperparameter | Value |
|----------------|-------|
| Vocabulary Size | 10,000 |
| Hidden Dimension (`d_model`) | 448 |
| Number of Transformer Blocks | 8 |
| Attention Heads (`n_heads`) | 7 |
| Head Dimension (`d_k`) | 64 |
| FFN Hidden Dimension (`d_ff`) | 2048 |
| Context Length | 512 tokens |
| Positional Representation | Learned absolute positional embeddings |
| Normalization | Pre-LayerNorm |
| Activation Function | GELU |
| Attention Mechanism | Standard multi-head self-attention with causal mask |
| Dropout | 0.1 |
| Weight Tying | Yes (LM head shares weights with token embedding) |

## 3. Transformer Block Structure
The model uses standard Pre-LayerNorm residual blocks:
1. `x = x + causal_self_attention(LayerNorm(x))`
2. `x = x + feed_forward(LayerNorm(x))`

After the 8 blocks, there is a final LayerNorm before the LM head:
`x = final_LayerNorm(x)`
`logits = LM_head(x)`

## 4. Positional Representation
We use learned absolute positional embeddings of shape `[context_length, d_model]`, which is `[512, 448]`.

## 5. Attention Dimensions
- `d_model` = 448
- `n_heads` = 7
- `d_k` = `d_model` / `n_heads` = 64

## 6. FFN Dimensions
The feed-forward network consists of two linear projections:
- Input projection: `d_model` (448) → `d_ff` (2048)
- Output projection: `d_ff` (2048) → `d_model` (448)

## 7. Weight Tying
Weight tying is applied between the token embedding matrix and the language modeling (LM) head. This means the LM head is a linear projection from `d_model` to `vocab_size` that uses the transposed token embedding weights, saving approximately 4.48 million parameters.

## 8. Parameter-Count Derivation

**Decision on Biases**: We explicitly use linear biases in all standard Linear projections (Q, K, V, output projection, and both FFN projections). The LM head has a tied weight matrix (shared with the token embedding) and an independent vocabulary-sized bias vector.

- **Token embedding**: `V × d_model` = 10,000 × 448 = 4,480,000
- **Position embedding**: `context_length × d_model` = 512 × 448 = 229,376

**Per Attention Block**:
- **Q projection**: `(d_model × d_model) + d_model` = (448 × 448) + 448 = 201,152
- **K projection**: 201,152
- **V projection**: 201,152
- **Output projection**: 201,152
- *Total Attention params*: 4 × 201,152 = 804,608

**Per FFN Block**:
- **Input projection**: `(d_model × d_ff) + d_ff` = (448 × 2048) + 2048 = 919,552
- **Output projection**: `(d_ff × d_model) + d_model` = (2048 × 448) + 448 = 917,952
- *Total FFN params*: 919,552 + 917,952 = 1,837,504

**Two LayerNorms per Transformer Block**:
- Each LN has a scale and shift vector of size `d_model`.
- *Total LN params per block*: 2 × (2 × 448) = 1,792

**Total Parameters per Transformer Block**: 
804,608 (Attention) + 1,837,504 (FFN) + 1,792 (LayerNorms) = 2,643,904
**All 8 Transformer Blocks**: 
8 × 2,643,904 = 21,151,232

**Final LayerNorm**:
- 2 × 448 = 896

**LM Head**:
- Weight: NO additional weight matrix because of weight tying (0 additional weight parameters)
- Bias: independent vocabulary-sized bias vector = 10,000

## 9. Expected Parameter Count
**Total Theoretical Parameters**: 
4,480,000 + 229,376 + 21,151,232 + 896 + 10,000 = **25,871,504**

This count has been verified programmatically from the instantiated PyTorch model using:
`sum(p.numel() for p in model.parameters() if p.requires_grad)`

## 10. Explanation of Why This Architecture Was Selected
1. **Head Dimension**: `d_model=448` and 7 attention heads gives an exact, clean head dimension of 64.
2. **Depth**: Eight layers provide meaningful depth for representations while keeping the model small enough for rapid experimentation and training on available hardware (RTX 3060).
3. **Feed-Forward Capacity**: `d_ff=2048` provides a substantial feed-forward component without making the model unnecessarily large.
4. **Vocabulary Size**: Fixed at 10,000 because both final Phase 1 tokenizers are strictly 10K BPE tokenizers.
5. **Positional Encoding**: Learned positional embeddings make the positional representation explicit and very simple to implement from scratch for the assignment.
6. **Pre-LayerNorm**: Used because it is a straightforward stable Transformer-block formulation, improving convergence predictability.
7. **Weight Tying**: Reduces unnecessary parameters between the input embedding and output vocabulary projection, ensuring the ~25M parameter budget is efficiently spent on the model depth rather than duplicated vocab embeddings.
8. **Parameter Reporting**: The exact parameter count will be programmatically derived from the actual PyTorch model rather than approximated in the final report.

## 11. Hindi vs Nepali Configuration Differences
While the codebase and architecture remain structurally identical, the configurations define strict data boundaries:
- **Hindi Model (Model H)**:
  - Tokenizer: `hindi/tokenizer/models/hindi_bpe_10k_final.model`
  - Corpus: `hindi/data/splits/`
- **Nepali Model (Model L)**:
  - Tokenizer: `nepali/tokenizer/models/nepali_bpe_10k_assignment.model` (The experimental `nepali_bpe_10k_final.model` MUST NOT be used)
  - Corpus: `nepali/data/final/`

## 12. Model Independence
**Explicit Statement:** The two models (Model H and Model L) possess completely independent parameter weights, separate embeddings, and individual checkpoints. There is zero transfer learning or parameter sharing between the Hindi and Nepali instantiations.
