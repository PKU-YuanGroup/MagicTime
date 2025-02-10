if [[ ! -d "logs" ]]; then
  mkdir logs
fi


job_name=${1-infer_cogvideox}
echo 'start job:' ${job_name}
now=$(date +"%Y%m%d_%H%M%S")
LOG_FILE=logs/${job_name}_${now}.log
echo 'log file: ' ${LOG_FILE}


# i2v
export CUDA_VISIBLE_DEVICES=1
python inference/test_cogvideo_t2v.py \
    --qwen_path /storage/ysh/Ckpts/Qwen2.5-VL-7B-Instruct/ \
    --model_path /storage/ysh/Ckpts/CogVideoX1.5-5B \
    --output_path samples/test \
    --num_frames 81 \
    --fps 16 \
    --seed 42 \
    2>&1 | tee ${LOG_FILE}