if [[ ! -d "logs" ]]; then
  mkdir logs
fi

export TOKENIZERS_PARALLELISM="false"
export WANDB_MODE="offline"
export WANDB_MODE="online"
export WANDB_API_KEY="81e90f25899751c659b19091506e48c18966af1b"

CKPT="/storage/ysh/Ckpts/CogVideoX1.5-5B"
output_dir="/storage/ysh/Code/MultiID/1_Code/MagicTime_Rebuttal/MagicTime_ongoing/train_demo_output"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc-per-node=8 --nnodes=1 --master-addr=localhost --master-port=12345 train.py \
  --csv_path /storage/ysh/Ckpts/ChronoMagic/caption/ChronoMagic_train.csv \
  --video_folder /storage/ysh/Ckpts/ChronoMagic/video \
  --pretrained_model_name_or_path $CKPT \
  --output_dir $output_dir \
  --dataloader_num_workers 8 \
  --pin_memory \
  --seed 42 \
  --mixed_precision bf16 \
  --train_batch_size 1 \
  --max_train_steps 5000 \
  --checkpointing_steps 282 \
  --gradient_accumulation_steps 1 \
  --learning_rate "5e-5" \
  --lr_scheduler "cosine_with_restarts" \
  --lr_warmup_steps 100 \
  --lr_num_cycles 1 \
  --optimizer "adamw" \
  --beta1 0.9 \
  --beta2 0.95 \
  --weight_decay 0.001 \
  --max_grad_norm 1.0 \
  --allow_tf32 \
  --report_to wandb \
  --nccl_timeout 1800 \
  --resume_from_checkpoint latest \
  --wandb_project cogvideox-magictime \
  --wandb_name cogvideox-magictime_train \
  --ema_decay 0.999 \
  --ema_interval 10 \
  2>&1 | tee ${LOG_FILE}

  # --enable_tiling \
  # --enable_slicing \