# Simpler: SupervIsed Multi-view rePresentation LEaRning for Math Word Problems

We introduce a novel multi-view score function for weighted contrastive learning that quantifies similarities between math problems, effectively enhancing fine-grained problem representation and understanding, and achieving state-of-the-art performance on three real-world MWP datasets.
## Environment Setup

This project uses Python 3.10. Follow the steps below to set up the project environment:

### Installing Dependencies

First, clone the repository and navigate to the project directory:

```bash
conda create --name simpler python=3.10
conda activate simpler
cd PROJECT_ROOT_PATH
pip install -r requirements.txt
```
### Training from Scratch
```bash
python main.py --train --dataset Math23k --similarity TLWD 
```
### Training LLM with Simpler
```bash
cd ./LLM_Simpler
python train_llm_cl.py
```
### Loading Pretrained Weights
Download pretrained Math23k sota ckpt from https://drive.google.com/file/d/1QuqAWW29Ael18wVeppgQ_XomrLdTgPuA/view?usp=sharing
```bash
python main.py --test --ckpt PATH_TO_CKPT
```
