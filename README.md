# Decentralized Policy Learning for Multi-Agent Reinforcement Learning

Research code for learning decentralized execution policies by distilling a
centralized Multi-Agent Transformer (MAT) teacher. This repository accompanies
the research project *Distributed Policy Learning in Multi-Agent Reinforcement
Learning*.

## Overview

Multi-agent systems must often coordinate under communication and scalability
constraints. A centralized policy can use the observations of every agent to
make coordinated decisions, but it may be impractical to deploy. A
decentralized policy can run from each agent's local observation, but typically
has less information and can lose performance.

This project learns a decentralized student policy from a centralized MAT
teacher. The objective is to retain the teacher's coordinated behavior while
making execution possible when agents only have local observations.

## Method

- A centralized MAT teacher is trained to produce coordinated multi-agent
  actions.
- Each student actor receives only its own observation and can therefore run
  independently at execution time.
- During training, student actors learn from the teacher's action distribution
  through a KL-divergence loss and from task rewards through a reinforcement
  learning loss.
- A student critic uses joint observations during training, following the
  centralized-training, decentralized-execution (CTDE) setting.

## Results

The figures below report the results used in the accompanying thesis. Each
curve summarizes ten training seeds and compares the MAT teacher, the
distilled student, MAPPO, and HAPPO. These are fixed research results, not
continuously updated CI benchmarks.

| Google Research Football: `academy_counterattack_easy` | SMAC: `3s5z_vs_3s6z` |
| --- | --- |
| <img src="images/research/academy_counterattack_easy.png" alt="Evaluation-score learning curves for Google Research Football academy_counterattack_easy" width="100%"> | <img src="images/research/3s5z_vs_3s6z.png" alt="Evaluation-score learning curves for SMAC 3s5z_vs_3s6z" width="100%"> |

## Reproducing experiments

The repository provides per-environment Docker images, YAML experiment
manifests, local GPU selection, and Slurm/Apptainer launch paths.

Initialize the required simulator submodules first:

```bash
git submodule update --init --recursive
```

Inspect a SMAC experiment before launching it:

```bash
uv run python mat/scripts/run_yaml.py mat/scripts/configs/smac/smac_single.yaml --dry-run
```

Detailed setup and launch instructions:

- [Docker environments and known limitations](docker/README.md)
- [YAML experiment configuration, sweeps, and Slurm submission](mat/scripts/configs/README.md)

Local results, checkpoints, W&B artifacts, Slurm output, and `.env` files are
excluded from version control.

## Repository layout

- `mat/algorithms/`: MAT, student-policy, and training implementations.
- `mat/envs/`: supported multi-agent environments.
- `mat/scripts/configs/`: versioned YAML experiment manifests.
- `docker/`: per-environment Dockerfiles and locked dependencies.
- `images/research/`: figures reported above.

## Relationship to the upstream project

This repository was forked from
[PKU-MARL/Multi-Agent-Transformer](https://github.com/PKU-MARL/Multi-Agent-Transformer).
It uses the upstream MAT implementation as the centralized teacher and adds
the decentralized student-policy research code, experiment configurations, and
reproducibility infrastructure. It is not the official MAT implementation.

## Upstream citation

Please cite the original MAT work when using the upstream algorithm:

```bibtex
@article{wen2022multi,
  title={Multi-Agent Reinforcement Learning is a Sequence Modeling Problem},
  author={Wen, Muning and Kuba, Jakub Grudzien and Lin, Runji and Zhang, Weinan and Wen, Ying and Wang, Jun and Yang, Yaodong},
  journal={arXiv preprint arXiv:2205.14953},
  year={2022}
}
```
