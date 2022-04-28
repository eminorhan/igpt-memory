#!/bin/bash

#SBATCH --account=cds
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=2
#SBATCH --mem=300GB
#SBATCH --time=00:20:00
#SBATCH --array=0
#SBATCH --job-name=finetune_igpt
#SBATCH --output=finetune_igpt_%A_%a.out

### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=16

module purge
module load cuda/11.3.1

LR=0.0005
OPTIMIZER='Adam'

srun python -u /scratch/eo41/minGPT/finetune.py '/scratch/eo41/konkle_1/study' --data_cache '/scratch/eo41/minGPT/data_model_cache/konkle_1_study_saycam.pth' --save_dir '/scratch/eo41/minGPT/data_model_cache' --epochs 25 --batch_size 2 --optimizer $OPTIMIZER --lr $LR --seed $SLURM_ARRAY_TASK_ID --n_layer 24 --n_head 8 --n_emb 512 --resume 'iGPT-S-SAYCam' --pretrain_data 'say' --finetune_data 'konkle_1_study'

echo "Done"
