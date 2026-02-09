#!/bin/bash

NGPUS=${1:-4}
LOGFILE=${2:-"logs_rs.txt"}

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# OMP_NUM_THREADS=16 \
# MKL_NUM_THREADS=1 \
# TORCH_CUDNN_SDPA_ENABLED=1 \
# torchrun --nproc_per_node="${NGPUS}" --master_port=29500 \
#   sam3_concept_bank_time.py \
#   --users "loveda=configs/cfg_loveda.py,
#            potsdam=configs/cfg_potsdam.py,
#            vaihingen=configs/cfg_vaihingen.py,
#            isaid=configs/cfg_isaid.py" \
#   --split train \
#   --checkpoint_path pretrained/sam3/sam3.pt \
#   --bpe_path pretrained/clip/bpe_simple_vocab_16e6.txt.gz \
#   --pad_ratio 0.05 \
#   --tau_w 0.15 \
#   --bg_thr_mode dice \
#   --output_pt "./configs/concept_bank/cb_sam3_rs.pt"

DATASETS=(
    loveda
    potsdam
    vaihingen
    isaid
)

: > "${LOGFILE}"

for BENCHMARK in "${DATASETS[@]}"; do
  {
    printf "\n=== DATASET: %s ===\n\n" \
      "${BENCHMARK}"
  } >> "${LOGFILE}"

  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  OMP_NUM_THREADS=16 \
  MKL_NUM_THREADS=1 \
  TORCH_CUDNN_SDPA_ENABLED=1 \
  torchrun --nproc_per_node="${NGPUS}" --master_port=29500 \
    eval.py \
    --launcher pytorch \
    --config "./configs/cfg_${BENCHMARK}.py" \
    |& tee -a "${LOGFILE}"

  echo "----------" >> "${LOGFILE}"
done

python3 summarize_seg_metrics.py "${LOGFILE}"
