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
git clone https://github.com/your-username/hssfla.git
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
