#!/bin/bash

SIGMAS=("0.01" "0.05")
BATCH_SIZES=("16" "32" "64" "128")

for sigma in "${SIGMAS[@]}"; do
  for batch_size in "${BATCH_SIZES[@]}"; do

    JOB_NAME="train_sigma${sigma}_bs${batch_size}"
    OUT="/neurospin/dico/babdelghani/Runs/02_champollion_v1/Output/yaware_volume/output/Large_cingulate_right/output_sigma${sigma}_bs${batch_size}.txt"
    ERR="/neurospin/dico/babdelghani/Runs/02_champollion_v1/Output/yaware_volume/error/Large_cingulate_right/error_sigma${sigma}_bs${batch_size}.txt"

    qsub -N "$JOB_NAME" \
         -o "$OUT" \
         -e "$ERR" \
         -v SIGMA=${sigma},BATCH_SIZE=${batch_size} \
         SC_left_train.pbs

  done
done
