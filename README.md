# CoboCLR

This code is derived from https://github.com/Levigty/AimCLR-v2, which is the implementation of the paper **"Improving Self-Supervised Action Recognition from Extremely Augmented Skeleton Sequences"**. 


## Data Preparation
- From CoboGesture dataset, acquire Annotation_v4 (.csv files) and pose_new_v2 (.npy files) -> distribute the continuous data into action segment with corresponding label with `cobot_per_action.py`.
- Perform quality assessment with `clean_low_quality_actions.py`.
- Perform missing joint recovery with `cleaning.py`.
- Prepare data for training and testing with `cleaned_to_cobot.py`

## Installation
  ```bash
# Install torchlight
$ cd torchlight
$ python setup.py install
$ cd ..
  
# Install other python libraries
$ pip install -r requirements.txt
  ```

## Unsupervised Pre-Training

Example for unsupervised pre-training of **CoboCLR**. You can change some settings of `.yaml` files in `config/three-stream/pretext` folder.
```bash

# train on CoboGesture xsub for three-stream
$ python main.py pretrain_aimclr_v2_3views --config config/three-stream/pretext/pretext_aimclr_v2_3views_cobot_xsub.yaml
```

## Linear Evaluation

Example for linear evaluation of **CoboCLR**. You can change `.yaml` files in `config/three-stream/linear` folder.
```bash
$ python main.py linear_evaluation --config config/three-stream/linear/linear_eval_aimclr_v2_3views_cobot_xsub.yaml
```

## Full finetune

Example for full finetuning of **CoboCLR**. You can change `.yaml` files in `config/three-stream/finetune` folder.
```bash
$ python main.py linear_evaluation --config config/three-stream/finetune/finetune_aimclr_v2_3views_cobot_xsub.yaml
```

## Semi-supervised

Example for semi-supervised of **CoboCLR**. You can change `semi.yaml` files in `config/three-stream/finetune` folder.
```bash
$ python main.py linear_evaluation --config config/three-stream/finetune/finetune_aimclr_v2_3views_cobot_xsub_semi.yaml
```


## Citation
```
@inproceedings{guo2022aimclr,
  Title= {Contrastive Learning from Extremely Augmented Skeleton Sequences for Self-supervised Action Recognition},
  Author= {Tianyu, Guo and Hong, Liu and Zhan, Chen and Mengyuan, Liu and Tao, Wang  and Runwei, Ding},
  Booktitle= {AAAI},
  Year= {2022}
}
```
