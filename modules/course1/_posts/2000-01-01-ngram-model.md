---
title: The N-Gram Model and Markov Chains
---

## Module 1.1: The N-Gram Model and Markov Chains

### Introduction to N-Gram Models

An **n-gram** is a contiguous sequence of *n* items from a given text or speech sequence. N-gram models are probabilistic language models that predict the next word in a sequence based on the previous words.

### The Markov Assumption

The fundamental principle behind n-gram models is the Markov assumption: the probability of a word depends only on a fixed number of previous words. For a bigram model (2-gram), we approximate:

$$P(w_n|w_{1:n-1}) \approx P(w_n|w_{n-1})$$

This simplification makes the model computationally tractable while still capturing local word dependencies.

### Computing N-Gram Probabilities

To compute n-gram probabilities, we use maximum likelihood estimation from a training corpus:

$$P(w_n|w_{n-1}) = \frac{C(w_{n-1}, w_n)}{C(w_{n-1})}$$

Where:
* $$C(w_{n-1}, w_n)$$ is the count of times words $$w_{n-1}$$ and $$w_n$$ appear together
* $$C(w_{n-1})$$ is the count of times word $$w_{n-1}$$ appears

### Higher-Order N-Grams

For trigram models (3-grams) and beyond:

$$P(w_n|w_{1:n-1}) \approx P(w_n|w_{n-2:n-1})$$

Higher-order n-grams capture more context but require more training data and storage.

### Smoothing Techniques

Real-world n-gram models require smoothing to handle unseen word combinations:

* **Laplace Smoothing**: Add a small constant to all counts
* **Kneser-Ney Smoothing**: Uses lower-order distributions for unseen n-grams
* **Backoff Models**: Fall back to lower-order n-grams when higher-order ones are unavailable

### Applications

N-gram models are foundational in:
* Speech recognition
* Machine translation
* Text generation
* Spelling correction
* Next-word prediction
