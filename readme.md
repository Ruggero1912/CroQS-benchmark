# Maybe you are looking for CroQS 🐊
## Cross-modal Query Suggestion for Text-to-Image Retrieval

![cross-modal query suggestion architecture schema](./images/cross-modal-query-suggestion-architecture.webp)

### Abstract

Query suggestion, a technique widely adopted in information retrieval, enhances system interactivity and the browsing experience of document collections.
In cross-modal retrieval, many works have focused on retrieving relevant items from natural language queries, while few have explored query suggestion solutions. 

In this work, we address query suggestion in cross-modal retrieval, introducing a novel task that focuses on suggesting minimal textual modifications needed to explore visually consistent subsets of the collection, following the premise of &ldquo;Maybe you are looking for&rdquo;.
To facilitate the evaluation and development of methods, we present a comprehensive benchmark named CroQS.
This dataset comprises initial queries, grouped result sets, and human-defined suggested queries for each group.
We establish dedicated metrics to rigorously evaluate the performance of various methods on this task, measuring representativeness, cluster specificity, and similarity of the suggested queries to the original ones.
Baseline methods from related fields, such as image captioning and content summarization, are adapted for this task to provide reference performance scores.

Although rather far from human performance, our experiments reveal that both LLM-based and captioning-based methods achieve competitive results on CroQS, improving the recall on cluster specificity by more than 122% and representativeness mAP by more than 23% with respect to the initial query.

---

## Repo content

In this repository you can find:
- the CroQS dataset as a json file
- the CroQS python class, which is the main entrypoint for benchmark usage
- an implementation of the set of baseline methods (ClipCap, DeCap and GroupCap)
- a couple of Jupyter Notebooks, [one](./benchmark-examples.ipynb) that report an usage example of the CroQS class to explore the dataset, and [the other](./evaluation.ipynb) that shows how to run evaluation experiments through it


## Setup


#### Prerequisites

- CUDA driver (check if everything works by typing `nvidia-smi`)

#### Steps

In order to run the code of this repo, 

1. create a new virtual environment (recommended):
`conda create --name croqs python==3.8` and then activate it `conda activate croqs`
2. clone the repository and `cd` into it
3. install the dependencies in requirements.txt: `pip3 install -r requirements.txt`
4. download coco dataset
5. create a `.env` file from the `.env.example` and update it with the real paths to coco dataset
