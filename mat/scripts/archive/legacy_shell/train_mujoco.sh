#!/bin/sh


#SBATCH -p dgx-a100-40g          # Partition
#SBATCH -G 1                     # Number of GPU
#SBATCH -t 1-0                   # Set time limit (1day) *
#SBATCH -J run                   # Job name
#SBATCH --mail-type=ALL          # when you want to get notifications. You can select one from [BEGIN , END , FAIL , REQUEUE , ALL] 
#SBATCH --mail-user=matsukawa@mi.t.u-tokyo.ac.jp # Email address which receives notifications
#SBATCH -o dump/stdout.%J             # stdout file name. %J is the job number.
#SBATCH -e dump/stderr.%J             # stderro file name. %J is the job number.

env="mujoco"
scenario="HalfCheetah-v2"
agent_conf="6x1"
agent_obsk=0
faulty_node=-1
#eval_faulty_node="-1 0 1 2 3 4 5"
eval_faulty_node="-1"
algo="mat"
exp="single"
seed=1
num_env_steps=10000000*2
final_noise_rate=0
eval_noise_rate=0.5
user_name="matsukawa-naoki555-university-of-tokyo"

export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"
echo "MUJOCO_PY_MUJOCO_PATH=${MUJOCO_PY_MUJOCO_PATH}"
export LD_LIBRARY_PATH="$MUJOCO_PY_MUJOCO_PATH/bin:$LD_LIBRARY_PATH"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

echo "env=${env}, scenario=${scenario}, algo=${algo}, exp=${exp}, seed=${seed}"

CUDA_VISIBLE_DEVICES=0 \
python train/train_mujoco.py \
  --seed "${seed}" \
  --env_name "${env}" \
  --algorithm_name "${algo}" \
  --experiment_name "${exp}" \
  --scenario "${scenario}" \
  --agent_conf "${agent_conf}" \
  --agent_obsk "${agent_obsk}" \
  --faulty_node "${faulty_node}" \
  --eval_faulty_node "${eval_faulty_node}" \
  --critic_lr 5e-5 --lr 5e-5 \
  --entropy_coef 0.001 --max_grad_norm 0.5 \
  --eval_episodes 5 --n_training_threads 16 \
  --n_rollout_threads 40 --num_mini_batch 40 \
  --episode_length 100 --eval_interval 25 \
  --num_env_steps "${num_env_steps}" --ppo_epoch 10 --clip_param 0.05 \
  --use_eval --add_center_xy --use_state_agent \
  --use_value_active_masks --use_policy_active_masks \
  --user_name "${user_name}" \
    --final_noise_rate "${final_noise_rate}" \
    --eval_noise_rate "${eval_noise_rate}" \
  --gradual
