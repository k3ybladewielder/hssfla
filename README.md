# A Generic Extractive Multi-document Text Summarization Method Using Memetic Algorithm and Combinatorial Optimization
This repo contains the code of the Holistic Text Summarization with the Shuffled Frog-Leaping Algorithm (HSSFLA) proposed on the paper A Generic Extractive Multi-document Text Summarization Method Using Memetic Algorithm and Combinatorial Optimization.

## Abstract
**Research Context**:Automatic text summarization remains a subject of considerable relevance across multiple domains. In particular, extractive multi-document generic summarization has garnered increased attention due to its capacity to mitigate information overload in a wide range of applications. **Scientific and/or Practical Problem**: The volume of unstructured text data produced on the internet has grown exponentially in recent years, driven by advances in information and communication technologies (ICTs). This massive generation of data makes it difficult for users to find relevant information. **Proposed Solution and/or Analysis**: This study introduces, implements, and applies the memetic algorithm known as Holistic Text Summarization with the Shuffled Frog-Leaping Algorithm (HSSFLA) to address the generic extractive multi-document multi-language text summarization problem using combinatorial optimization techniques. **Related IS Theory**: This research integrates swarm intelligence, memetic algorithms and combinatorial optimization. **Research Method**: An in vitro experiment was conducted to quantitatively compare the summary quality between the proposed method and similar methods in the literature. **Summary of Results**: Experiments were carried out on the DUC2001/2002 benchmark datasets, and performance was evaluated using the Recall-Oriented Understudy for Gisting Evaluation (ROUGE) metric. The results demonstrate that the proposed approach yielded an average improvement of 25.12% in ROUGE-1 and 34.91% in ROUGE-2 on the DUC 2001 dataset. On the DUC2002 dataset, the method achieved average gains of 35.42% in ROUGE-1 and 36.08% in ROUGE-2. **Contributions and Impact to IS area**: HSSFLA, a memetic algorithm based on swarm intelligence, was developed to solve this problem for the first time. It creates holistic summaries, in which it evaluates the quality of the summary as a whole, rather than focusing exhaustively on finding the best individual sentences. HSSFLA outperforms the results of the scientific literature in DUC2001 and DUC2002.

# How to Use
This project uses **uv** to manage the virtual environment and dependencies.

### Requirements

- Python 3.9+
- uv installed

Install uv:

```bash
pip install uv
````

or

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Setup

Clone the repository:

```bash
git clone https://github.com/k3ybladewielder/hssfla.git
cd hssfla
```

Create the virtual environment:

```bash
uv venv
```

Activate it:

* macOS / Linux

```bash
source .venv/bin/activate
```

* Windows (PowerShell)

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
uv pip install -r requirements.txt
```

### Tutorial

This tutorial presents a minimal example of how to use the **HSSFLA (Holistic Text Summarization with the Shuffled Frog-Leaping Algorithm)** to generate an extractive summary from a simple text.

#### 0. Importing required libraries

First, import the necessary functions from the project modules:

```python
from preprocessing import process_corpus_with_stemming
from hssfla import hssfla
```

#### 1. Input text definition

In this example, a fictitious *Lorem Ipsum* text is used only to demonstrate the method’s workflow. In a real scenario, this text may correspond to a single document or to the concatenation of multiple documents.

```python
corpus = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit. 
Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. 
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.
Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
"""
```

#### 2. Text preprocessing

The text is segmented into sentences and transformed into vector representations (*embeddings*).
The preprocessing step includes normalization, tokenization, and stemming.

```python
embeddings, preprocessed_sentences_list = process_corpus_with_stemming(
    corpus,
    use_max_pooling=False
)
```

#### 3. HSSFLA parameter configuration

The main hyperparameters of the algorithm are defined as follows:

```python
n_cycle = 10        # Number of global cycles
pop_size = 200      # Population size
n_iter = 25         # Number of iterations per memeplex
n_memeplex = 5      # Number of memeplexes
gamma = 0.5         # Mixing rate
```

Additional important parameters include:

* `beta_min` and `beta_max`: control the intensity of the local search
* `L`: maximum summary length (in sentences)
* `epsilon`: small value to avoid division by zero

#### 4. Running the summarization algorithm

The HSSFLA is then executed to find the best subset of sentences that maximizes the overall summary quality.

```python
best_global_individual, best_global_fitness, best_sentences = hssfla(
    embeddings,
    preprocessed_sentences_list,
    n_cycle=n_cycle,
    n_iter=n_iter,
    n_memeplex=n_memeplex,
    beta_max=0.5,
    beta_min=0.1,
    L=L,
    pop_size=pop_size,
    gamma=gamma,
    epsilon=1e-8
)
```

* `best_global_individual`: binary solution indicating the selected sentences
* `best_global_fitness`: fitness value of the best solution
* `best_sentences`: list of sentences chosen for the summary

#### 5. Final summary generation

The selected sentences are concatenated to form the final extractive summary:

```python
candidate_summary = " ".join(best_sentences)
print(candidate_summary)
```

The output is an extractive summary that preserves the original sentences, selected in a **holistic** manner by jointly considering relevance, redundancy, and information coverage.

---

💡 **Note**:
For multi-document summarization, simply concatenate all documents into a single string (`corpus`) before the preprocessing step.


# How to Cite
```
@INPROCEEDINGS{248295,
    AUTHOR="Alysson Guimarães and Methanias Colaço Júnior",
    TITLE="A Generic Extractive Multi-document Text Summarization Method Using Memetic Algorithm and Combinatorial Optimization",
    BOOKTITLE="SBSI 2026 - TP-SI () ",
    ADDRESS="",
    DAYS="18-21",
    MONTH="may",
    YEAR="2026",
    ABSTRACT="Research Context:Automatic text summarization remains a subject of considerable relevance across multiple domains. In particular, extractive multi-document generic summarization has garnered increased attention due to its capacity to mitigate information overload in a wide range of applications. Scientific and/or Practical Problem: The volume of unstructured text data produced on the internet has grown exponentially in recent years, driven by advances in information and communication technologies (ICTs). This massive generation of data makes it difficult for users to find relevant information. Proposed Solution and/or Analysis: This study introduces, implements, and applies the memetic algorithm known as Holistic Text Summarization with the Shuffled Frog-Leaping Algorithm (HSSFLA) to address the generic extractive multi-document multi-language text summarization problem using combinatorial optimization techniques. Related IS Theory: This research integrates swarm intelligence, memetic algorithms and combinatorial optimization. Research Method: An in vitro experiment was conducted to quantitatively compare the summary quality between the proposed method and similar methods in the literature. Summary of Results: Experiments were carried out on the DUC2001/2002 benchmark datasets, and performance was evaluated using the Recall-Oriented Understudy for Gisting Evaluation (ROUGE) metric. The results demonstrate that the proposed approach yielded an average improvement of 25.12% in ROUGE-1 and 34.91% in ROUGE-2 on the DUC 2001 dataset. On the DUC2002 dataset, the method achieved average gains of 35.42% in ROUGE-1 and 36.08% in ROUGE-2. Contributions and Impact to IS area: HSSFLA, a memetic algorithm based on swarm intelligence, was developed to solve this problem for the first time. It creates holistic summaries, in which it evaluates the quality of the summary as a whole, rather than focusing exhaustively on finding the best individual sentences. HSSFLA outperforms the results of the scientific literature in DUC2001 and DUC2002.",
    KEYWORDS="Inteligência artificial (generativa, LLM, PLN, entre outros) em sistemas de informação; Sistemas de informação para gestão de dados, informação e conhecimento",
    URL="http://XXXXX/248295.pdf"
}
```
