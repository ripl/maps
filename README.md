# MAPS: Max-aggregation Active Policy Selection

Official code for "Active Policy Improvement from Multiple Black-box Oracles":

[Xuefeng Liu](https://www.linkedin.com/in/xuefeng-liu-658278200/),* [Takuma Yoneda](https://takuma.yoneda.xyz),* [Chaoqi Wang](https://alecwangcq.github.io),* [Matthew R. Walter](https://ttic.edu/walter), and [Yuxin Chen](https://yuxinchen.org), "Active Policy Improvement from Multiple Black-box Oracles", in Proceedings of the International Conference on Machine Learning (ICML), 2023 (* denotes equal contribution) [[arXiv]](https://arxiv.org/abs/2306.10259)

# Preparing a runtime environment
You can either build a docker image based on `docker/Dockerfile`, or pull from dockerhub: `docker pull ripl/maps`

# Usage
The current script uses weights and biases. You may need to set `WANDB_API_KEY` environment variable.

### 1. Pretrain experts with `maps/scripts/pretraining/train_expert.py`  
For example, from the project root directory, you can run:
```bash
$ python3 -m maps.scripts.pretraining.train_expert sac model_dir --env-name dmc:Cheetah-run-v1
```
This trains a SAC policy on `Cheetah-run` env, and saves the network weights periodically under `model_dir`.

### 2. Create a sweep file.  
The following command creates multiple run configurations over environment domains, set of experts and algorithms

\* Before running the following, you need to adjust the expert paths (L6 in `maps/scripts/pretraining/experts.py`) to the one you saved expert models to in the previous step.
```bash
$ python3 -m maps.scripts.sweep.sample_sweep
```
This generates `sample_sweep.jsonl`

  
### 3. Run training by specifying a line number of the sweep file.  
 For example, to launch the configuration in the first line:
 ```bash
 $ python3 -m maps.scripts.train maps/scripts/sweep/sample_sweep.jsonl -l 0
 ```


# TODOs
- [ ] Remove the Dockerfile's dependency on Takuma's image
- [ ] Push the new docker image to dockerhub
- [ ] Make the pretrained experts available? (git lfs?)


## Citing MAPS

If you find our work useful in your research, please consider citing the paper as follows:

``` bibtex
@inproceedings{liu23,
    Author    = {Xuefeng Liu and Takuma Yoneda and Chaoqi Wang and Matthew R. Walter and Yuxin Chen},
    Title     = {Active Policy Improvement from Multiple Black-box Oracles},
    Booktitle = {Proceedings of the International Conference on Machine Learning (ICML)},
    Year      = {2023},
}
```
