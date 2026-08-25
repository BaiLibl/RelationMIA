# 2026-06-20

for f in outputs/*uf*.pkl; do
    python cal_mia.py --input "$f"
done

# cal_store_run() {
#     CUDA_VISIBLE_DEVICES=$1 python cal_store.py \
#         --model_name qwen-0.5b \
#         --sft_data $2 \
#         --dpo_data $2 \
#         --n_member 1000 \
#         --n_nonmember 1000 \
#         --cal_model $3 \
#         --batch_size 8 \
#         --output outputs/cal_score_${2}_${2}_qwen-0.5b_$(basename "$3").pkl \
#         > logs/cal_store_${2}_$(basename "$3").log 2>&1
# }

# mkdir -p logs
# mkdir -p outputs

# cal_store_run 4 uf Qwen/Qwen2.5-0.5B-Instruct &
# cal_store_run 6 uf Qwen/Qwen2.5-1.5B-Instruct &
# wait 

# cal_store_run 4 uf TinyLlama/TinyLlama-1.1B-Chat-v1.0 &
# cal_store_run 6 uf HuggingFaceTB/SmolLM2-1.7B-Instruct &
# wait 

# cal_store_run 4 uf microsoft/Phi-3-mini-4k-instruct &
# cal_store_run 6 se Qwen/Qwen2.5-0.5B-Instruct &
# wait 

# cal_store_run 4 se Qwen/Qwen2.5-1.5B-Instruct &
# cal_store_run 6 se TinyLlama/TinyLlama-1.1B-Chat-v1.0 &
# wait 

# cal_store_run 4 se HuggingFaceTB/SmolLM2-1.7B-Instruct &
# cal_store_run 6 se microsoft/Phi-3-mini-4k-instruct &
# wait 


# cal_store_run 6 uf meta-llama/Llama-3.2-1B-Instruct
# cal_store_run 6 uf google/gemma-2-2b-it
# wait

# 2026-6-22
# cal_bg_run() {
#     mkdir -p bg_outputs

#     CUDA_VISIBLE_DEVICES=$1 python cal_bg_store.py \
#         --model_name qwen-0.5b \
#         --sft_data $2 \
#         --dpo_data $2 \
#         --cal_model $3 \
#         --bg_split train \
#         --bg_start 1000 \
#         --n_background 5000 \
#         --batch_size 8 \
#         --output bg_outputs/cal_bg_${2}_qwen-0.5b_$(basename "$3").pkl \
#         > logs/cal_bg_${2}_$(basename "$3").log 2>&1
# }

# cal_bg_run 4 uf Qwen/Qwen2.5-0.5B-Instruct &
# cal_bg_run 6 uf Qwen/Qwen2.5-1.5B-Instruct &
# cal_bg_run 4 uf TinyLlama/TinyLlama-1.1B-Chat-v1.0 &
# cal_bg_run 6 uf HuggingFaceTB/SmolLM2-1.7B-Instruct &
# cal_bg_run 4 uf microsoft/Phi-3-mini-4k-instruct &


# cal_bg_run 6 se Qwen/Qwen2.5-0.5B-Instruct &
# wait 

# cal_bg_run 4 se Qwen/Qwen2.5-1.5B-Instruct &
# cal_bg_run 6 se TinyLlama/TinyLlama-1.1B-Chat-v1.0 &
# wait 

# cal_bg_run 4 se HuggingFaceTB/SmolLM2-1.7B-Instruct &
# cal_bg_run 6 se microsoft/Phi-3-mini-4k-instruct &
# wait 

