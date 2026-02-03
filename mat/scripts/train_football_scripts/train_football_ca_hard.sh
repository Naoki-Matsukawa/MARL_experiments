#!/bin/sh
# exp param
env="Football"
scenario="academy_counterattack_hard"
algo="rmappo" # "mappo" "ippo"
exp="check"

seed=1
student_kl_coef=1.0
student_rl_coef=1.0
student_rl_coef_start=0.0

user_name="matsukawa-naoki555-university-of-tokyo"

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"

export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -nr | head -1 | awk -F', ' '{print $2}')
echo "Using GPU device: $CUDA_VISIBLE_DEVICES"
echo "coef kl: ${student_kl_coef}, coef rl: ${student_rl_coef}"

# football param
num_agents=4

# train param
num_env_steps=50000000
episode_length=1000

echo "n_rollout_threads: ${n_rollout_threads} \t ppo_epoch: ${ppo_epoch} \t num_mini_batch: ${num_mini_batch}"

uv run ../train/train_football.py \
 --env_name ${env} --scenario_name ${scenario} --algorithm_name ${algo} --experiment_name ${exp} --seed ${seed} \
 --num_agents ${num_agents} --num_env_steps ${num_env_steps} --episode_length ${episode_length} \
 --representation "simple115v2" --rewards "scoring,checkpoints" --n_rollout_threads 50 --ppo_epoch 15 --num_mini_batch 2 \
 --save_interval 200000 --log_interval 200000 --use_eval --eval_interval 400000 --n_eval_rollout_threads 100 --eval_episodes 100 \

