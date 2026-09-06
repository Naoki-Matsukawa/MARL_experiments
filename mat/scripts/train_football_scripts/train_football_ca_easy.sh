#!/bin/sh
# exp param
env="football"
scenario="academy_counterattack_easy"
algo="r_mappo" # "mappo" "ippo"
exp="check"
exp="single"

seed=1

# football param
num_agents=4

user_name="matsukawa-naoki555-university-of-tokyo"
# train param
num_env_steps=25000000
episode_length=200
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -nr | head -1 | awk -F', ' '{print $2}')
echo "Using GPU device: $CUDA_VISIBLE_DEVICES"
echo "n_rollout_threads: ${n_rollout_threads} \t ppo_epoch: ${ppo_epoch} \t num_mini_batch: ${num_mini_batch}"

export PYTHONPATH=~/Multi-Agent-Transformer:$PYTHONPATH

uv run ../train/train_football.py \
 --env_name ${env} --scenario_name ${scenario} --algorithm_name ${algo} --experiment_name ${exp} --seed ${seed} \
 --num_agents ${num_agents} --num_env_steps ${num_env_steps} --episode_length ${episode_length} \
 --representation "simple115v2" --rewards "scoring,checkpoints" --n_rollout_threads 50 --ppo_epoch 15 --num_mini_batch 2 \
 --save_interval 200000 --log_interval 200000 --use_eval --eval_interval 400000 --n_eval_rollout_threads 100 --eval_episodes 100 \
 --user_name ${user_name}
