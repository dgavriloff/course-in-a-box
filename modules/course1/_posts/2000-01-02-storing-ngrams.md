---
title: Storing and Querying N-Gram Models
---

## Module 1.2: Storing and Querying N-Gram Models

### The Storage Challenge

N-gram models can become extremely large. For a vocabulary of size *V* and n-gram order *n*, a naive storage approach would require $$V^n$$ entries. For practical applications with vocabularies of 100,000+ words, efficient storage is crucial.

### Trie Data Structure

A **trie** (prefix tree) is the most common data structure for storing n-grams efficiently:

* Shares prefixes among n-grams
* Reduces redundant storage
* Enables efficient prefix-based lookups
* Typical space savings: 50-80% compared to flat storage

#### Trie Structure Example

For bigrams like "the cat", "the dog", "a cat":
```
root
├── "the"
│   ├── "cat" [probability]
│   └── "dog" [probability]
└── "a"
    └── "cat" [probability]
```

### ARPA Format

The ARPA format is a standard text-based format for storing n-gram language models:

```
\data\
ngram 1=50000
ngram 2=250000
ngram 3=500000

\1-grams:
-1.5 the -0.3
-2.1 cat -0.2
...

\2-grams:
-0.8 the cat -0.1
-1.2 the dog -0.15
...

\3-grams:
-0.5 the black cat
...

\end\
```

#### ARPA Format Components

* **Probability**: Stored as $$\log_{10}$$ of the n-gram probability
* **Backoff Weight**: Weight used when backing off to lower-order n-grams
* **N-gram Entry**: Context words followed by the predicted word

### Binary Formats

For production systems, binary formats offer:
* Faster loading times
* Smaller file sizes
* Memory-mapped access
* Common formats: KenLM binary, SRILM binary

### Quantization and Compression

Modern n-gram models use additional compression:

* **Probability Quantization**: Store probabilities with reduced precision (e.g., 8-bit instead of 32-bit)
* **Huffman Coding**: Variable-length encoding for frequent n-grams
* **Delta Encoding**: Store differences between adjacent probabilities

### Querying Strategies

Efficient querying requires:

1. **Hash-based Lookup**: Fast O(1) average-case lookup for exact n-grams
2. **Backoff Traversal**: Navigate to lower-order n-grams when higher-order not found
3. **Caching**: Keep frequently accessed n-grams in memory

### Trade-offs

Storage and query optimization involves balancing:
* **Model Size** vs **Accuracy**: Pruning rare n-grams saves space but reduces coverage
* **Load Time** vs **Query Speed**: Memory-mapped files are slower to initialize but faster to query
* **Compression** vs **Access Speed**: Highly compressed models require decompression overhead
