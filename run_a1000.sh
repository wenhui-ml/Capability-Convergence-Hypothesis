#!/bin/bash
# Amendment 11: A-ii deciding cells at n=1000, GPU 3 only.
cd .
python common/launch.py \
  --jobs logs/jobs_a1000.txt --gpus 3 --per_gpu 2 \
  --log_dir logs/run_a1000 >> logs/a1000.log 2>&1
touch logs/A1000_DONE
