---
title: Evaluation of Language Generation
---

## Module 1.3: Evaluation of Language Generation

### Intrinsic vs Extrinsic Evaluation

Language models can be evaluated in two ways:

* **Intrinsic Evaluation**: Measures the model's internal performance (perplexity, likelihood)
* **Extrinsic Evaluation**: Measures performance on downstream tasks (translation, summarization)

### Perplexity

**Perplexity** is the most common intrinsic metric for language models. It measures how well a probability model predicts a sample:

$$PP(W) = P(w_1, w_2, ..., w_N)^{-\frac{1}{N}}$$

Or equivalently:

$$PP(W) = \sqrt[N]{\frac{1}{P(w_1, w_2, ..., w_N)}}$$

#### Interpreting Perplexity

* Lower perplexity indicates better prediction
* Perplexity of *X* means the model is as uncertain as if it had to choose uniformly from *X* possibilities
* Typical values: 50-200 for good models on standard benchmarks

### BLEU Score

**BLEU** (Bilingual Evaluation Understudy) measures the quality of machine-generated text against reference translations:

$$BLEU = BP \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

Where:
* $$p_n$$ is the precision for n-grams of length *n*
* $$BP$$ is a brevity penalty for short outputs
* Typically computed for n=1 to n=4

#### BLEU Characteristics

* Range: 0 to 1 (or 0 to 100 when expressed as percentage)
* Focuses on precision (what percentage of generated words appear in references)
* Widely used in machine translation
* Does not account for semantic similarity or grammatical correctness

### ROUGE Score

**ROUGE** (Recall-Oriented Understudy for Gisting Evaluation) measures overlap between generated and reference summaries:

* **ROUGE-N**: N-gram overlap (similar to BLEU but recall-focused)
* **ROUGE-L**: Longest common subsequence
* **ROUGE-S**: Skip-bigram overlap

#### ROUGE Formula (ROUGE-N)

$$ROUGE\text{-}N = \frac{\sum_{S \in References} \sum_{gram_n \in S} Count_{match}(gram_n)}{\sum_{S \in References} \sum_{gram_n \in S} Count(gram_n)}$$

### Critique of Traditional Metrics

While perplexity, BLEU, and ROUGE have been foundational, they have significant limitations:

* **Lack of Semantic Understanding**: These metrics cannot assess whether generated text is meaningful or factually correct
* **Insensitivity to Word Order**: N-gram overlap ignores syntactic and discourse structure beyond local context
* **Poor Human Correlation**: Modern studies show weak correlation with human judgments of quality
* **Gaming the Metrics**: Models can be optimized to score well without improving actual generation quality
* **Multiple Valid Outputs**: These metrics struggle when many different phrasings are equally valid
* **No Creativity Assessment**: Cannot measure novelty or appropriateness of creative generations

### Modern Alternatives

Recent research has developed more sophisticated metrics:

* **BERTScore**: Uses contextual embeddings to measure semantic similarity
* **BLEURT**: Learned metric trained on human ratings
* **Human Evaluation**: Still considered the gold standard, using criteria like:
  * Fluency
  * Coherence
  * Relevance
  * Factual accuracy
  * Informativeness
