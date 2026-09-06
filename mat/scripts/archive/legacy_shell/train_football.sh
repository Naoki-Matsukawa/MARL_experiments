#!/bin/bash
#SBATCH -p dgx-a100-40g          # Partition
#SBATCH -G 1                     # Number of GPU
#SBATCH -t 1-0                   # Set time limit (1day) *
#SBATCH -J run                   # Job name
#SBATCH --mail-type=ALL          # when you want to get notifications. You can select one from [BEGIN , END , FAIL , REQUEUE , ALL] 
#SBATCH --mail-user=matsukawa@mi.t.u-tokyo.ac.jp # Email address which receives notifications
#SBATCH -o dump/stdout.%J             # stdout file name. %J is the job number.
#SBATCH -e dump/stderr.%J             # stderro file name. %J is the job number.



env="football"
#scenario="academy_counterattack_easy"
scenario="11_vs_11_hard_stochastic"
# scenario="11_vs_11_easy_stochastic"
# academy_pass_and_shoot_with_keeper
# academy_3_vs_1_with_keeper
# academy_counterattack_easy
scenario="academy_counterattack_easy"
n_agent=4
algo="mat"
# algo="r_mappo"
exp="single"
seed=5

student_kl_coef=1.0
student_rl_coef=1.0
student_rl_coef_start=0.0
student_aux_value_coef=0.0
user_name="matsukawa-naoki555-university-of-tokyo"

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"

export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -nr | head -1 | awk -F', ' '{print $2}')
echo "Using GPU device: $CUDA_VISIBLE_DEVICES"
echo "coef kl: ${student_kl_coef}, coef rl: ${student_rl_coef}"

seeds=(6 7 8 9 10)

for seed in "${seeds[@]}";
do
echo "Starting training with seed ${seed} ..."
python train/train_football.py \
 --seed ${seed} \
 --env_name ${env} \
 --algorithm_name ${algo} \
 --experiment_name ${exp} \
 --scenario ${scenario} \
 --n_agent ${n_agent} \
 --lr 5e-4 \
 --entropy_coef 0.01 \
 --max_grad_norm 0.5 \
 --eval_episodes 32 \
 --n_training_threads 16 \
 --n_rollout_threads 20 \
 --num_mini_batch 1 \
 --episode_length 200 \
 --eval_interval 25 \
 --num_env_steps 10000000 \
 --ppo_epoch 10 \
 --clip_param 0.05 \
 --use_eval \
 --use_value_active_masks \
 --use_policy_active_masks \
 --save_gifs \
 --distillation \
 --user_name ${user_name} \
 --student_kl_coef ${student_kl_coef} \
 --student_rl_coef ${student_rl_coef}\
 --student_rl_linear_schedule \
 --student_rl_coef_start ${student_rl_coef_start} \
 --max_grad_norm 5 \
 --student_aux_value_coef ${student_aux_value_coef} &


done
wait    

echo "All Python scripts have finished."

    