# RelationMIA

Reference implementation for PreMIA-style membership inference baselines on preference-learning data.

The current codebase intentionally focuses on the reference-based baseline:

```text
score_pair = (log p_dpo(y_w | x) - log p_dpo(y_l | x))
           - (log p_sft(y_w | x) - log p_sft(y_l | x))
```

This baseline is kept as the comparison point for future reference-free relation-level attacks.

## Entry points

- `train_sft_dpo.py`: train SFT and DPO LoRA adapters.
- `output_store.py`: cache target/reference logprobs to `.npy`.
- `output_mia.py`: evaluate cached PreMIA baseline scores.
- `mia_population.py`: run the same baseline directly without writing a cache file.

## Examples

```bash
python train_sft_dpo.py --model_name gpt2 --sft_data imdb --dpo_data imdb
python output_store.py --model_name gpt2 --sft_data imdb --dpo_data imdb
python output_mia.py --model_name gpt2 --sft_data imdb --dpo_data imdb
python mia_population.py --model_name gpt2 --sft_data imdb --dpo_data imdb --popu 10
```
