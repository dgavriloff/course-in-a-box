---
title: From N-Grams to Neural Networks
---

## Module 2.1: From N-Grams to Neural Networks

### The Limitations of N-Gram Models

While n-gram models served as the foundation of language modeling for decades, they have fundamental constraints:

* **Data Sparsity**: Most possible n-grams never appear in training data
* **Fixed Context**: Cannot adapt context window size dynamically
* **No Generalization**: Treating words as discrete symbols prevents learning semantic similarities
* **Storage Requirements**: Exponential growth with vocabulary size and n-gram order

### Neural Language Models

Neural networks address these limitations by learning continuous representations:

#### Word Embeddings

Instead of treating words as discrete symbols, neural models represent them as dense vectors in continuous space:

* Words with similar meanings have similar vectors
* Semantic relationships are captured (e.g., king - man + woman ≈ queen)
* Typical dimensions: 100-1000

### Recurrent Neural Networks (RNNs)

RNNs were the first successful neural approach to language modeling:

* Process sequences one token at a time
* Maintain hidden state capturing context
* Can theoretically handle arbitrary-length dependencies

#### LSTM and GRU

Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU) improved RNNs:

* Gates control information flow
* Better at capturing long-range dependencies
* Reduced vanishing gradient problems

### The Transformer Revolution

The **Transformer architecture** (2017) revolutionized language modeling:

#### Self-Attention Mechanism

The core innovation is self-attention, which computes relationships between all tokens simultaneously:

* Parallel processing of entire sequences
* Direct modeling of dependencies regardless of distance
* Multiple attention heads capture different relationships

#### Key Components

1. **Multi-Head Attention**: Multiple attention mechanisms run in parallel
2. **Position Encodings**: Add position information to embeddings
3. **Feed-Forward Networks**: Process each position independently
4. **Layer Normalization**: Stabilize training
5. **Residual Connections**: Enable deep architectures

### From Transformers to Large Language Models

Modern LLMs build on the Transformer foundation:

* **GPT Series**: Decoder-only architecture for generation
* **BERT**: Encoder-only for understanding tasks
* **T5/BART**: Encoder-decoder for sequence-to-sequence

#### Scaling Laws

Research has shown that model performance scales predictably with:

* Model size (number of parameters)
* Dataset size
* Compute budget

This insight drove the development of increasingly large models:
* GPT-2 (2019): 1.5B parameters
* GPT-3 (2020): 175B parameters
* GPT-4 (2023): Estimated 1.7T parameters

### Emergent Capabilities

As models scale, they exhibit emergent abilities not present in smaller models:

* Few-shot learning
* Chain-of-thought reasoning
* Tool use and function calling
* Cross-lingual transfer

### Key Advantages Over N-Grams

Neural models, especially Transformers, surpass n-grams in:

* **Generalization**: Learn semantic patterns, not just surface statistics
* **Context**: Attention mechanisms handle long-range dependencies
* **Efficiency**: Continuous representations are more compact
* **Adaptability**: Fine-tuning enables task-specific optimization
* **Multimodal**: Can extend to images, audio, and other modalities
