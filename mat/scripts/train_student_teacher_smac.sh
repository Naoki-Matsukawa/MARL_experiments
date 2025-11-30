#!/bin/sh

#SBATCH -p dgx-a100-40g          # Partition
#SBATCH -G 1                     # Number of GPU
#SBATCH -t 1-0                   # Set time limit (1day) *
#SBATCH -J run                   # Job name
#SBATCH --mail-type=ALL          # when you want to get notifications. You can select one from [BEGIN , END , FAIL , REQUEUE , ALL] 
#SBATCH --mail-user=matsukawa@mi.t.u-tokyo.ac.jp # Email address which receives notifications
#SBATCH -o dump/stdout.%J             # stdout file name. %J is the job number.
#SBATCH -e dump/stderr.%J             # stderro file name. %J is the job number.


env="StarCraft2"
map="6h_vs_8z"
algo="mat"
exp="single"
seed=1


student_kl_coef=1.0
student_rl_coef=1.0

user_name="matsukawa-naoki555-university-of-tokyo"

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -nr | head -1 | awk -F', ' '{print $2}')
echo "Using GPU device: $CUDA_VISIBLE_DEVICES"
echo "coef kl: ${student_kl_coef}, coef rl: ${student_rl_coef}"

python train/train_smac.py --env_name ${env}\
 --algorithm_name ${algo} \
 --experiment_name ${exp} \
 --map_name ${map} \
 --seed ${seed} \
 --n_training_threads 16 \
 --n_rollout_threads 4 \
 --num_mini_batch 1 \
 --episode_length 100 \
 --num_env_steps 10000000 \
 --lr 5e-4 --ppo_epoch 15 \
 --clip_param 0.05 \
 --save_interval 100000 \
 --use_value_active_masks \
 --use_eval  \
 --distillation \
 --user_name ${user_name} \
 --student_kl_coef ${student_kl_coef} \
 --student_rl_coef ${student_rl_coef}
