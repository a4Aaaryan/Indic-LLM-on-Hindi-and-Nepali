# Phase 2 Report: Pretraining and Evaluation


## 1. Experimental Setup

- **Models**: Separate Hindi and Nepali decoder-only Transformer models.
- **Architecture**: 8 layers, 448 d_model, 7 heads (64 head dim), 2048 d_ff, 512 context length.
- **Parameter Count**: 25871504 (approx 25.8M) per model.
- **Tokenizer & Corpus**: Independent SentencePiece BPE 10K tokenizers trained from scratch on independent corpora. Hindi uses `hindi_bpe_10k_final.model`, Nepali uses `nepali_bpe_10k_assignment.model`.
- **Checkpoint Used for Evaluation**: Step 20000 (20,000).
- **Decoding Strategies**: Greedy, Temperature 0.5, Temperature 1.0, and Temperature 1.5.

## 2. Intrinsic Evaluation

Intrinsic metrics evaluate the model's ability to predict the next token on unseen validation data.

| Metric | Hindi (Validation) | Nepali (Validation) |
|---|---|---|
| **Validation Loss** | 4.1015 | 4.1790 |
| **Perplexity** | 60.4292 | 65.2973 |
| **Bits Per Byte (BPB)** | 0.7017 | 0.5435 |

**Comparison**: Hindi performs slightly better on validation loss and perplexity, but Nepali exhibits a lower Bits Per Byte (BPB). The lower BPB in Nepali suggests better compression per raw text byte, which aligns with Nepali having slightly lower fertility and a different corpus composition.

## 3. Generation Evaluation

Generation metrics evaluate the quality of text generated given a prompt. ROUGE-L is omitted from substantive interpretation (see Section 5).

### Hindi Generation Metrics

| Strategy | BLEU | chrF | Repetition Rate | Distinct-1 | Distinct-2 |
|---|---|---|---|---|---|
| **greedy** | 4.1595 | 18.9905 | 0.8020 | 0.3652 | 0.4445 |
| **temp_0.5** | 4.9206 | 20.8806 | 0.6196 | 0.5931 | 0.7725 |
| **temp_1.0** | 5.1846 | 20.5922 | 0.1934 | 0.8912 | 0.9952 |
| **temp_1.5** | 4.5572 | 21.2614 | 0.1189 | 0.9336 | 0.9951 |

### Nepali Generation Metrics

| Strategy | BLEU | chrF | Repetition Rate | Distinct-1 | Distinct-2 |
|---|---|---|---|---|---|
| **greedy** | 5.3712 | 21.0384 | 0.7307 | 0.4007 | 0.4501 |
| **temp_0.5** | 5.2468 | 23.6248 | 0.6351 | 0.5487 | 0.6441 |
| **temp_1.0** | 3.5921 | 26.4760 | 0.0950 | 0.9402 | 0.9918 |
| **temp_1.5** | 3.2390 | 23.8434 | 0.0457 | 0.9771 | 1.0000 |

**Temperature Effects**: Lower temperatures (greedy, 0.5) show higher repetition rates and lower distinct-N scores, meaning they suffer from degenerate repeating loops. As temperature increases (1.0, 1.5), repetition drastically decreases and distinct-N approaches 1.0 (higher diversity). However, BLEU drops significantly at temperature 1.5 as the generation becomes too random to match the reference text.

## 4. Attention Analysis

| Metric | Hindi | Nepali |
|---|---|---|
| **Layer 0 Mean Entropy** | 1.9417 | 2.2250 |
| **Layer 0 Mean Distance** | 6.0668 | 6.6935 |
| **Layer 7 Mean Entropy** | 1.7901 | 1.7572 |
| **Layer 7 Mean Distance** | 8.1937 | 8.0824 |

**Early vs. Late Layers**: In both models, early layers (Layer 0) exhibit higher mean distance and lower entropy, indicating sharp, localized attention focusing heavily on specific earlier tokens (e.g., positional patterns). Late layers (Layer 7) display higher entropy and lower distance, showing dispersed, broad attention that focuses primarily on immediately preceding context to synthesize final output distributions.

**Hindi vs. Nepali Comparison**: The attention behavior is structurally identical across both models, suggesting the Transformer successfully learned the same core language-modeling mechanisms regardless of corpus variations.

## 5. ROUGE-L Limitation

The computed ROUGE-L score is exactly 0.0000 across all decoding strategies for both languages. **This metric is excluded from substantive interpretation** and cannot be used as a valid performance comparison. The standard `rouge_score` library utilizes an English-centric regex tokenizer (`[a-z0-9]+`) that aggressively strips all Devanagari characters. Consequently, it parses 0 tokens from both the generated and reference Indic strings, producing an artificial zero score.

## 6. Limitations

- **Training Duration**: The models were evaluated at 20,000 steps. While validation loss had stabilized, longer training might yield further qualitative improvements.
- **Independent Models**: The models were trained completely independently. A shared multilingual model with a unified vocabulary might have yielded cross-lingual transfer benefits, especially for a lower-resource language like Nepali.
- **Evaluation Metrics**: Standard NLP metrics like ROUGE-L fail on Indic scripts out-of-the-box due to Latin-centric tokenization assumptions. Future evaluations require custom metric tokenizers.
