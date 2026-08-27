python mia_generated_sim.py \
  --model_name qwen-0.5b \
  --sft_data uf \
  --dpo_data uf \
  --n_member 1000 \
  --n_nonmember 1000 \
  --num_responses 8 \
  --topk 2 \
  --output outputs/generated_sim_uf_uf_qwen-0.5b_k8.pkl


# python mia_generated_sim.py \
#   --input outputs/generated_sim_uf_uf_qwen-0.5b_k8.pkl \
#   --topk 2

# python mia_generated_sim.py \
#   --input outputs/generated_sim_uf_uf_qwen-0.5b_k8.pkl \
#   --sim_filter 0.3