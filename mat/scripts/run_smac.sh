#!/bin/bash

#SBATCH -p dgx-a100-40g          # Partition
#SBATCH -G 1                     # Number of GPU
#SBATCH -t 1-0                   # Set time limit (1day) *
#SBATCH -J run                   # Job name
#SBATCH --mail-type=ALL          # when you want to get notifications. You can select one from [BEGIN , END , FAIL , REQUEUE , ALL] 
#SBATCH --mail-user=matsukawa@mi.t.u-tokyo.ac.jp # Email address which receives notifications
#SBATCH -o dump/stdout.%J             # stdout file name. %J is the job number.
#SBATCH -e dump/stderr.%J             # stderro file name. %J is the job number.

source /home/mil/matsukawa/Multi-Agent-Transformer/.venv/bin/activate
env="StarCraft2"
map="6h_vs_8z"


# map="27m_vs_30m"

map="3s5z_vs_3s6z"


# algo="r_mappo"
algo="mat"
#algo="happo"
exp="single"

seed_start=0
seed_count=5

student_kl_coef=1.0
student_rl_coef=1.0
# student_rl_coef=0.0
student_rl_coef_start=0.0

user_name="matsukawa-naoki555-university-of-tokyo"

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -nr | head -1 | awk -F', ' '{print $2}')
echo "Using GPU device: $CUDA_VISIBLE_DEVICES"
echo "coef kl: ${student_kl_coef}, coef rl: ${student_rl_coef}"
model_dir="/home/mil/matsukawa/Multi-Agent-Transformer/mat/scripts/results/StarCraft2/3s5z_vs_3s6z/mat/single/wandb/run-20260115_185536-34utly1d/files/transformer_9374.pt"


# CUDA_VERSION=11.8 
# export CUDA_HOME=/usr/local/cuda-$CUDA_VERSION
episode_length=100
clip_param=0.2
ppo_epoch=5
num_env_steps=10000000
gain=0.01
entropy_coef=0.01

noise_std=0.1
noise_std=0

gamma=0.99

if [ "$map" = "3s5z_vs_3s6z" ]; then
    echo "Using 3s5z_vs_3s6z "
    ppo_epoch=5
    clip_param=0.05
    num_env_steps=20000000
    
    if [ "$algo" = "r_mappo" ]; then
    echo "Using R-MAPPO "
    clip_param=0.2
    fi

    if [ "$algo" = "happo" ]; then
        echo "Using HAPPO "
        clip_param=0.2
        gamma=0.95
    fi

    if [ "$algo" = "mat" ]; then 
        echo "Using MAT "
        episode_length=100
    fi
    
fi
num_env_steps=30000000
eval_episodes=1000
seeds=(1)

for seed in "${seeds[@]}";
do
echo "Starting training with seed ${seed} ..."
python train/train_smac.py --env_name ${env}\
 --algorithm_name ${algo} \
 --experiment_name ${exp} \
 --map_name ${map} \
 --seed ${seed} \
 --n_training_threads 16 \
 --n_rollout_threads 32 \
 --num_mini_batch 1 \
 --episode_length ${episode_length} \
 --num_env_steps ${num_env_steps} \
 --lr 5e-4 \
 --ppo_epoch ${ppo_epoch} \
 --clip_param ${clip_param} \
 --save_interval 100000 \
 --gain ${gain} \
 --entropy_coef ${entropy_coef} \
 --gamma ${gamma} \
 --use_value_active_masks \
 --use_eval  \
 --distillation \
 --user_name ${user_name} \
 --student_kl_coef ${student_kl_coef} \
 --student_rl_coef ${student_rl_coef}\
 --student_rl_linear_schedule \
 --student_rl_coef_start ${student_rl_coef_start} \
 --max_grad_norm 5 \
 --noise_std ${noise_std}\
 --model_dir ${model_dir} \
 --eval_episodes ${eval_episodes} \
  &
done
wait

echo "All Python scripts have finished."
