#!/bin/sh

#SBATCH -p dgx-a100-40g          # Partition
#SBATCH -G 1                     # Number of GPU
#SBATCH -t 1-0                   # Set time limit (1day) *
#SBATCH -J run                   # Job name
#SBATCH --mail-type=ALL          # when you want to get notifications. You can select one from [BEGIN , END , FAIL , REQUEUE , ALL] 
#SBATCH --mail-user=matsukawa@mi.t.u-tokyo.ac.jp # Email address which receives notifications
#SBATCH -o dump/stdout.%J             # stdout file name. %J is the job number.
#SBATCH -e dump/stderr.%J             # stderro file name. %J is the job number.

# 共有ストレージがマウントされている計算ノードを 1 台だけ探す
echo "=== 基本情報 ==="
echo "HOSTNAME   : $(hostname)"
echo "SLURM_NODE : $SLURMD_NODENAME"
echo "DATE       : $(date)"

echo -e "\n=== マウント状況 (NFS など) ==="
# よく使われる共有ディレクトリをフィルタ
mount | grep -E '/home|/apps|/opt|lustre|bee|nfs' || echo "match nothing"

echo -e "\n=== ディスク使用量 ==="
df -hT | head -n 20      # 全部見たい場合は head を外す

echo -e "\n=== モジュール一覧 ==="
# module コマンドが無効ならスキップ
type module &>/dev/null && module list || echo "module command not found"

echo -e "\n=== PATH / LD_LIBRARY_PATH ==="
echo $PATH | tr ':' '\n' | head
echo "---"
echo ${LD_LIBRARY_PATH:-"(empty)"} | tr ':' '\n' | head

echo -e "\n=== Python / コンパイラのバージョン ==="
python -V 2>/dev/null || echo "python not found"
gcc --version | head -n 1 2>/dev/null || echo "gcc not found"