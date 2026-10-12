
# bash run3.sh > log3.txt 2>&1

#2026-10-10
# CUDA_VISIBLE_DEVICES=1 python factor_store.py \
#   --model_name qwen-0.5b \
#   --sft_data uf \
#   --dpo_data uf \
#   --n_member 1000 \
#   --n_nonmember 1000 \
#   --embed_model sentence-transformers/all-mpnet-base-v2 \
#   --embed_batch_size 64 \
#   --output outputs/factor_uf_uf_qwen-0.5b.npy

CUDA_VISIBLE_DEVICES=1 python factor_store.py \
  --model_name qwen-0.5b \
  --sft_data se \
  --dpo_data se \
  --n_member 1000 \
  --n_nonmember 1000 \
  --embed_model sentence-transformers/all-mpnet-base-v2 \
  --embed_batch_size 64 \
  --output outputs/factor_se_se_qwen-0.5b.npy

# python npy_to_csv.py --input outputs/factor_uf_uf_qwen-0.5b.npy --output factor_uf_uf_qwen-0.5b.csv
# python npy_to_csv.py --input outputs/factor_se_se_qwen-0.5b.npy --output factor_se_se_qwen-0.5b.csv

# 2026-9-14
# CUDA_VISIBLE_DEVICES=1 python mia_sim.py \
#   --model_name qwen-0.5b \
#   --sft_data se \
#   --dpo_data se \
#   --n_member 1000 \
#   --n_nonmember 1000 \
#   --num_return_sequences 4 \
#   --output response/sim_se_se_qwen-0.5b_k8.npy


# CUDA_VISIBLE_DEVICES=2 python mia_response.py \
#   --model_name qwen-0.5b \
#   --sft_data se \
#   --dpo_data se \
#   --n_member 1000 \
#   --n_nonmember 1000 \
#   --num_return_sequences 2 \
#   --output response/premia_uf_uf_qwen-0.5b.npy


# python mia_generated_sim.py \
#   --model_name qwen-0.5b \
#   --sft_data uf \
#   --dpo_data uf \
#   --n_member 1000 \
#   --n_nonmember 1000 \
#   --num_responses 8 \
#   --topk 2 \
#   --output outputs/generated_sim_uf_uf_qwen-0.5b_k8.pkl


# python mia_generated_sim.py \
#   --input outputs/generated_sim_uf_uf_qwen-0.5b_k8.pkl \
#   --topk 2

# python mia_generated_sim.py \
#   --input outputs/generated_sim_uf_uf_qwen-0.5b_k8.pkl \
#   --sim_filter 0.3