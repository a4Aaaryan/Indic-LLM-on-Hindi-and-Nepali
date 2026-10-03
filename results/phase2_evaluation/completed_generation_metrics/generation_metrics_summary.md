# Generation Metrics Summary

## Methodology

- **BLEU**: Sentence-level BLEU via `sacrebleu.sentence_bleu`. Internal tokenization (no external tokenizer).

- **chrF**: Sentence-level chrF via `sacrebleu.sentence_chrf`. Character n-gram F-score.

- **ROUGE-L**: Longest common subsequence F-measure via `rouge_score.RougeScorer` (no stemmer).

- **Distinct-1/Distinct-2**: Unique unigrams/bigrams divided by total unigrams/bigrams on whitespace-tokenized text.

- **Repetition Rate**: Fraction of tokens in the generated text that appear more than once.

- **References**: Gold continuations extracted from the corpus during generation, saved in `generation_samples.json`.

- **Tokenization**: BLEU and chrF use sacrebleu internal tokenization. Distinct-N and repetition use whitespace splitting. Both methods handle Hindi and Nepali Unicode identically.


## Per-Strategy Results

### Hindi

**greedy**: BLEU=4.16, chrF=18.99, ROUGE-L=0.0000, rep=0.8020, D1=0.3652, D2=0.4445

**temp_0.5**: BLEU=4.92, chrF=20.88, ROUGE-L=0.0000, rep=0.6196, D1=0.5931, D2=0.7725

**temp_1.0**: BLEU=5.18, chrF=20.59, ROUGE-L=0.0000, rep=0.1934, D1=0.8912, D2=0.9952

**temp_1.5**: BLEU=4.56, chrF=21.26, ROUGE-L=0.0000, rep=0.1189, D1=0.9336, D2=0.9951



### Nepali

**greedy**: BLEU=5.37, chrF=21.04, ROUGE-L=0.0000, rep=0.7307, D1=0.4007, D2=0.4501

**temp_0.5**: BLEU=5.25, chrF=23.62, ROUGE-L=0.0000, rep=0.6351, D1=0.5487, D2=0.6441

**temp_1.0**: BLEU=3.59, chrF=26.48, ROUGE-L=0.0000, rep=0.0950, D1=0.9402, D2=0.9918

**temp_1.5**: BLEU=3.24, chrF=23.84, ROUGE-L=0.0000, rep=0.0457, D1=0.9771, D2=1.0000



## Best and Worst Strategies

### Hindi

- **bleu**: best=temp_1.0 (5.1846), worst=greedy (4.1595)

- **chrf**: best=temp_1.5 (21.2614), worst=greedy (18.9905)

- **rougeL**: best=greedy (0.0000), worst=greedy (0.0000)

- **distinct_1**: best=temp_1.5 (0.9336), worst=greedy (0.3652)

- **distinct_2**: best=temp_1.0 (0.9952), worst=greedy (0.4445)

- **repetition_rate** (lower is better): best=temp_1.5 (0.1189), worst=greedy (0.8020)



### Nepali

- **bleu**: best=greedy (5.3712), worst=temp_1.5 (3.2390)

- **chrf**: best=temp_1.0 (26.4760), worst=greedy (21.0384)

- **rougeL**: best=greedy (0.0000), worst=greedy (0.0000)

- **distinct_1**: best=temp_1.5 (0.9771), worst=greedy (0.4007)

- **distinct_2**: best=temp_1.5 (1.0000), worst=greedy (0.4501)

- **repetition_rate** (lower is better): best=temp_1.5 (0.0457), worst=greedy (0.7307)



## Hindi vs Nepali Comparison (Greedy)

- **bleu**: Hindi=4.1595, Nepali=5.3712

- **chrf**: Hindi=18.9905, Nepali=21.0384

- **rougeL**: Hindi=0.0000, Nepali=0.0000

- **repetition_rate**: Hindi=0.8020, Nepali=0.7307

- **distinct_1**: Hindi=0.3652, Nepali=0.4007

- **distinct_2**: Hindi=0.4445, Nepali=0.4501


## Unavailable Metrics

All six metrics (BLEU, chrF, ROUGE-L, repetition_rate, Distinct-1, Distinct-2) were successfully computed. No metrics are unavailable.
