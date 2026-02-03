#!/bin/sh

#SBATCH -p dgx-a100-40g          # Partition
#SBATCH -G 1                     # Number of GPU
#SBATCH -t 1-0                   # Set time limit (1day) *
#SBATCH -J run                   # Job name
#SBATCH --mail-type=ALL          # when you want to get notifications. You can select one from [BEGIN , END , FAIL , REQUEUE , ALL] 
#SBATCH --mail-user=matsukawa@mi.t.u-tokyo.ac.jp # Email address which receives notifications
#SBATCH -o dump/stdout.%J             # stdout file name. %J is the job number.
#SBATCH -e dump/stderr.%J             # stderro file name. %J is the job number.

# chmod +x ../../.venv/bin/activate
# source ../../.venv/bin/activate

env="MPE"
scenario="simple_spread"  # simple_speaker_listener # simple_reference simple_spread
 
num_landmarks=3
num_agents=10
algo="mat"
algo="r_mappo"
exp="single"
seed=1
num_env_steps=20000000
#num_env_steps=200
final_noise_rate=0
eval_noise_rate=0
user_name="matsukawa-naoki555-university-of-tokyo"



echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=0 python train/train_mpe.py \
 --env_name ${env} --algorithm_name ${algo} \
 --experiment_name ${exp} \
 --scenario_name ${scenario} \
 --num_agents ${num_agents}\
 --num_landmarks ${num_landmarks} \
 --seed ${seed} --n_block 1 --n_embd 64 \
 --n_training_threads 16 --n_rollout_threads 128 \
 --num_mini_batch 1 --episode_length 25 \
 --num_env_steps 20000000 --ppo_epoch 10 \
 --clip_param 0.05 --use_ReLU --gain 0.01 --lr 7e-4 \
 --critic_lr 7e-4 --use_eval \
 --user_name ${user_name} \
 --save_gifs --num_env_steps ${num_env_steps} \
 --final_noise_rate ${final_noise_rate} \
 --eval_noise_rate ${eval_noise_rate}\
  