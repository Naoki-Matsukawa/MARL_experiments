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
map="3s5z_vs_3s6z"
algo="pld"
exp="pld_offline"
seed=1

user_name="matsukawa-naoki555-university-of-tokyo"

pld_dataset_dir="/home/mil/matsukawa/Multi-Agent-Transformer/mat/algorithms/PLD/dataset"
pld_vae_path="/home/mil/matsukawa/Multi-Agent-Transformer/mat/algorithms/PLD/vae_ckpt.pt"
pld_updates=50

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -nr | head -1 | awk -F', ' '{print $2}')
echo "Using GPU device: $CUDA_VISIBLE_DEVICES"

python train/train_smac.py --env_name ${env}\
 --algorithm_name ${algo} \
 --experiment_name ${exp} \
 --map_name ${map} \
 --seed ${seed} \
 --n_training_threads 16 \
 --n_rollout_threads 1 \
 --n_eval_rollout_threads 1 \
 --num_mini_batch 1 \
 --episode_length 100 \
 --num_env_steps 10000000 \
 --use_value_active_masks \
 --use_eval \
 --user_name ${user_name} \
 --pld_dataset_dir ${pld_dataset_dir} \
 --pld_updates ${pld_updates} \
 --pld_vae_path ${pld_vae_path} \
 --pld_freeze_vae \
 --pld_disable_obs_encoder
