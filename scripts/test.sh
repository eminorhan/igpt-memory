#!/bin/bash

#SBATCH --account=cds
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200GB
#SBATCH --time=4:00:00
#SBATCH --array=0-71
#SBATCH --job-name=test_igpt
#SBATCH --output=test_igpt_%A_%a.out

module purge
module load cuda/11.3.1

CONDITIONS=(novel_seen novel_unseen 1_exemplar_seen 1_exemplar_unseen 2_exemplar_seen 2_exemplar_unseen 4_exemplar_seen 4_exemplar_unseen 8_exemplar_seen 8_exemplar_unseen 16_exemplar_seen 16_exemplar_unseen)
MODEL_DIRS=(saycam_1_0 saycam_1_1 imagenet_1_0 imagenet_1_1 tabularasa_1_0 tabularasa_1_1)
MODEL_DIR=${MODEL_DIRS[$SLURM_ARRAY_TASK_ID / 12]}
CONDITION="${CONDITIONS[$SLURM_ARRAY_TASK_ID % 12]}"

echo $MODEL_DIR
echo $CONDITION

python -u /scratch/eo41/minGPT/test.py /scratch/eo41/konkle_1/test/$CONDITION \
    --batch_size 25 \
    --d_img 64 \
    --model_dir /scratch/eo41/minGPT/konkle_2010_models/$MODEL_DIR \
    --save_name losses_on_test_${CONDITION}_${MODEL_DIR}

echo "Done"
