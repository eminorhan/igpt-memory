#!/bin/bash

#SBATCH --account=cds
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=generate_igpt
#SBATCH --output=generate_igpt_%A_%a.out

module purge
module load cuda/11.3.1

python -u /scratch/eo41/minGPT/generate.py --condition 'half' --data_cache '/scratch/eo41/minGPT/data_model_cache/saycam_train.pth' --model_cache 'iGPT-S-SAYCam.pt' --filename 'saycam_robot.pdf' --img_dir '/scratch/eo41/brenden_1/test/robot'

echo "Done"
