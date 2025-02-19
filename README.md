

# DORA

Code for DORA: [Debiased Offline Representation Learning for Fast Online Adaptation in Non-stationary Dynamics](https://icml.cc/virtual/2024/poster/34708).

## Installation
- Please refer to `requirements.txt` with python version of 3.8.13

  ```python
  pip install -r ./requirement.txt
  ```

## Usage
### Offline Datasets Collection

Please collect your offline datasets in `./envs`, which generates tasks with different dynamics in MuJoco. For each single-task dataset, you may use general online RL methods to train policies independently and then restore all the transitions into the replay buffer during training as the datasets. 

As demonstrated in our paper, we use Soft Actor Critic (SAC) to train RL policies in 10 tasks with different dynamics. And we gather 200,000 transitions from replay buffer for each single-task dataset, except for the tasks of Pendulum-gravity, which comprises 40,000 transitions. For example, our datasets for `Cheetah-gravity` can be found in [drive](https://drive.google.com/drive/folders/1NmzHmzNY_P-ianKOXtKYsOtf-UHo5HZK?usp=sharing).

### Encoder Training

An example:

```
python run_offline_encoder_trainer.py --env_type cheetah-gravity --rnn_fix_length 8 --varying_params gravity --seed 0
```
Parameters:
- `--env_type`: The possible environments includes `Cheetah-gravity, Cheetah-dof, Cheetah-body_mass, Hopper-gravity, Pendulum-gravity, Walker-gravity ` with changing dynamics named  `gravity, dof, body_mass ` . 
- `--rnn_fix_length`: The RNN history length. 
- `--varying_params`: The changing dynamics, e.g. `gravity `.
- `--seed`: Random seeds.

### Meta Policy Training

An example:

```
python run_offline_policy_trainer.py --env cheetah-dof --rnn_fix_length 8 --encoder_path ''  --seed 0
```

Parameters:

- `--encoder_path`: The path of the trained encoder model, e.g. `./log/env_type/your_experiment/model`.

## 

We express our gratitude to Fanmin Luo and Yihao Sun for their help in our code.

## Reference

- [ESCP](https://github.com/FanmingL/ESCP)
- [OfflineRL-Kit](https://github.com/yihaosun1124/OfflineRL-Kit)


## Citation

If you find our paper or code useful, please consider citing via

```latex
@inproceedings{dora,
  author       = {Xinyu Zhang and
                  Wenjie Qiu and
                  Yi{-}Chen Li and
                  Lei Yuan and
                  Chengxing Jia and
                  Zongzhang Zhang and
                  Yang Yu},
  title        = {Debiased Offline Representation Learning for Fast Online Adaptation
                  in Non-stationary Dynamics},
  booktitle    = {Forty-first International Conference on Machine Learning, {ICML} 2024,
                  Vienna, Austria, July 21-27, 2024},
  publisher    = {OpenReview.net},
  year         = {2024},
  url          = {https://openreview.net/forum?id=BrZPj9rEpN},
}
```
