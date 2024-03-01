# CoT-BERT : Enhancing Unsupervised Sentence Representation through Chain-of-Thought  

- Our paper: [[2309.11143\] CoT-BERT: Enhancing Unsupervised Sentence Representation through Chain-of-Thought (arxiv.org)](https://arxiv.org/abs/2309.11143)

  __We updated the paper on February 29, 2024, but this is still not the final version__

## Abstract

Unsupervised sentence representation learning aims to transform input sentences into fixed-length vectors enriched with intricate semantic information while obviating the reliance on labeled data. Recent progress within this field, propelled by contrastive learning and prompt engineering, has significantly bridged the gap between unsupervised and supervised strategies. Nonetheless, the potential utilization of Chain-of-Thought, remains largely untapped in this trajectory. To unlock the latent capabilities of pre-trained models, such as BERT, we propose a two-stage approach for sentence representation: comprehension and summarization. Subsequently, the output of the latter phase is harnessed as the embedding of the input sentence. For further performance enhancement, we introduce an extended InfoNCE Loss by incorporating the contrast between positive and negative instances. Additionally, we also refine the existing template denoising technique to better mitigate the perturbing influence of prompts on input sentences. Rigorous experimentation substantiates our method, CoT-BERT, transcending a suite of robust baselines without necessitating other text representation models or external databases.

## Results

|             Model             | STS12 | STS13 | STS14 | STS15 | STS16 | STSb  | SICK-R | Avg.  |
| :---------------------------: | :---: | :---: | :---: | :---: | :---: | :---: | :----: | :---: |
|  Unsupervised CoT-BERT-base   | 72.56 | 85.53 | 77.91 | 85.05 | 80.94 | 82.40 | 71.41  | 79.40 |
| Unsupervised CoT-RoBERTa-base | 75.43 | 85.47 | 78.74 | 85.64 | 82.21 | 83.40 | 73.46  | 80.62 |

- Our Checkpoints

  - CoT-BERT-base Checkpoint

    https://drive.google.com/file/d/1RjNr9QYlOIWn9SUcOqVd9jedWkGg5UVk/view?usp=sharing

  - CoT-RoBERTa-base Checkpoint

    https://drive.google.com/file/d/1B-vryHG6-DFuNBW_RZ_STqn7-bFnBicL/view?usp=sharing

## Setup

- Install Dependencies

  ```sh
  pip install -r requirements.txt
  ```

- Download Data

  ```sh
  cd SentEval/data/downstream/
  bash download_dataset.sh
  cd -
  cd ./data
  bash download_wiki.sh
  bash download_nli.sh
  cd -
  ```

- Train with BERT-base

  ```sh
  cd code-for-BERT
  python train.py
  ```

- Train with RoBERTa-base

  ```sh
  cd code-for-RoBERTa
  python train.py
  ```

- Python version 3.8.16

## Acknowledgement

- Our code is based on PromptBERT