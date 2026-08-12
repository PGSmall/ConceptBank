#!/bin/bash

NGPUS=${1:-4}
LOGFILE=${2:-"logs_ns.txt"}

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# OMP_NUM_THREADS=16 \
# MKL_NUM_THREADS=1 \
# TORCH_CUDNN_SDPA_ENABLED=1 \
# torchrun --nproc_per_node="${NGPUS}" --master_port=29500 \
#   sam3_concept_bank_time.py \
#   --users "voc21=configs/cfg_voc21.py,
#            context60=configs/cfg_context60.py,
#            coco_object=configs/cfg_coco_object.py,
#            voc20=configs/cfg_voc20.py,
#            context59=configs/cfg_context59.py,
#            coco_stuff164k=configs/cfg_coco_stuff164k.py,
#            city_scapes=configs/cfg_city_scapes.py,
#            ade20k=configs/cfg_ade20k.py" \
#   --split train \
#   --checkpoint_path sam3/assets/sam3.pt \
#   --bpe_path sam3/assets/bpe_simple_vocab_16e6.txt.gz \
#   --cand_topk 999 --fuse_topk 999 \
#   --confidence_threshold 0.1 \
#   --use_context_view --use_masked_view \
#   --output_pt "./configs/concept_bank/cb_sam3_ns.pt"

DATASETS=(
  voc21
  context60
  coco_object
  voc20
  context59
  coco_stuff164k
  city_scapes
  ade20k
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
