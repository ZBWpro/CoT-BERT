# CoT-BERT : Enhancing Unsupervised Sentence Representation through Chain-of-Thought  

- Our paper: [[2309.11143\] CoT-BERT: Enhancing Unsupervised Sentence Representation through Chain-of-Thought (arxiv.org)](https://arxiv.org/abs/2309.11143)

  __Our paper was updated on March 8, 2024, but it is not yet the final version.__

## Abstract

Unsupervised sentence representation learning endeavors to transform input sentences into fixed-length vectors enriched with intricate semantic information while obviating the reliance on labeled data. Recent strides in this domain have been significantly propelled by breakthroughs in contrastive learning and prompt engineering. Despite these advancements, the field has reached a plateau, leading some researchers to incorporate external components to enhance sentence embeddings' quality. Such integration, though beneficial, complicates the solutions and inflates the demand for computational resources. In response to these challenges, this paper presents CoT-BERT, an innovative method that harnesses the progressive thinking of Chain-of-Thought reasoning to tap into the latent potential of pre-trained models like BERT. Additionally, we develop an advanced contrastive learning loss function and propose a novel template denoising strategy. Rigorous experimentation substantiates CoT-BERT surpasses a range of well-established baselines by relying exclusively on the intrinsic strengths of pre-trained models. 

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