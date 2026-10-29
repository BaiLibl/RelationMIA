mkdir -p outputs


CUDA_VISIBLE_DEVICES=0 python output_user.py \
  --model_name qwen-0.5b \
  --sft_data prism \
  --dpo_data prism \
  --n_member 1000000 \
  --n_nonmember 1000000 \
  --output outputs/user_prism_qwen-0.5b.npy

# CUDA_VISIBLE_DEVICES=0 python output_store.py \
#   --model_name qwen-0.5b \
#   --sft_data prism \
#   --dpo_data prism \
#   --n_member 1000 \
#   --n_nonmember 1000 \
#   --output outputs/premia_prism_prism_qwen-0.5b.npy

# # 读取保存结果，计算 AUROC 和 TPR@FPR=0.05
# python output_mia.py \
#   --input outputs/premia_prism_prism_qwen-0.5b.npy \
#   --popu 1

# python output_mia.py \
#   --input outputs/premia_prism_prism_qwen-0.5b.npy \
#   --popu 5

# python train_sft_dpo.py \
#   --model_name qwen-0.5b \
#   --sft_data prism \
#   --dpo_data prism \