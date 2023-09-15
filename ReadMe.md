# CoT-BERT : Enhancing Unsupervised Sentence Representation through Chain-of-Thought  

## Abstract

Unsupervised sentence representation learning aims to transform input sentences into fixed-length vectors enriched with intricate semantic information while obviating the reliance on labeled data. Recent progress within this field, propelled by contrastive learning and prompt engineering, has significantly bridged the gap between unsupervised and supervised strategies. Nonetheless, the potential utilization of Chain-of-Thought, remains largely untapped within this trajectory. To unlock latent capabilities within pre-trained models, such as BERT, we propose a two-stage approach for sentence representation: comprehension and summarization. Subsequently, the output of the latter phase is harnessed as the vectorized representation of the input sentence. 

For further performance enhancement, we meticulously refine both the contrastive learning loss function and the template denoising technique for prompt engineering. Rigorous experimentation substantiates our method, CoT-BERT, transcending a suite of robust baselines without necessitating other text representation models or external databases.

## Results

|          Model           | STS12 | STS13 | STS14 | STS15 | STS16 | STSb  | SICK-R | Avg.  |
| :----------------------: | :---: | :---: | :---: | :---: | :---: | :---: | :----: | :---: |
|  Unsupervised CoT-BERT   | 72.56 | 85.53 | 77.91 | 85.05 | 80.94 | 82.40 | 71.41  | 79.40 |
| Unsupervised CoT-RoBERTa | 75.43 | 85.47 | 78.74 | 85.64 | 82.21 | 83.40 | 73.46  | 80.62 |

- Our Checkpoints

  - CoT-BERT Checkpoint

    https://drive.google.com/file/d/1cwSNuAw8EJnqIAHDOjMGMFFHMr0qdh2d/view?usp=drive_link

  - CoT-RoBERTa Checkpoint

    https://drive.google.com/file/d/1uEE2tVH1D5c4h4VPzRCT7Nf3HDr1_HZc/view?usp=drive_link

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

## Acknowledgement

- Our code is based on PromptBERT