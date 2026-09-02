# CUDA_VISIBLE_DEVICES=3 python mia_query.py \
#   --model_name qwen-0.5b \
#   --sft_data uf \
#   --dpo_data uf \
#   --n_member 1000 \
#   --n_nonmember 1000 \
#   --sem_generation_batch_size 4 \
#   --output query/query_uf_uf_qwen-0.5b.npy

# python query_postprocess.py \
#   --input query/query_uf_uf_qwen-0.5b.npy \
#   --output query/query_uf_uf_qwen-0.5b_post.npy

  CUDA_VISIBLE_DEVICES=1 python mia_query.py \
  --model_name qwen-0.5b \
  --sft_data uf \
  --dpo_data uf \
  --n_member 1000 \
  --n_nonmember 1000 \
  --sem_generation_batch_size 8 \
  --output query/query_abc_logps_uf_uf_qwen-0.5b.npy