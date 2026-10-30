import os


MODEL_DICT = {
    "gpt2": "gpt2",
    "gpt2-medium": "gpt2-medium",
    "gpt2-large": "gpt2-large",
    "llama-3b": "openlm-research/open_llama_3b",
    "qwen-0.5b": "Qwen/Qwen2.5-0.5B",
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B",
}


def default_split_sizes(dataset_name):
    name = dataset_name.lower()
    if name == "uf":
        return 50000, 10000
    if name == "se":
        return 80000, 10000
    if name == 'prism':
        return 29330, 11574 # train, eval
    return 10000, 10000


def model_run_name(model_name, sft_data, dpo_data):
    return f"{model_name}-{sft_data}-{dpo_data}"


def model_paths(model_name, sft_data, dpo_data, root="../models"):
    run_name = model_run_name(model_name, sft_data, dpo_data)
    return {
        "sft": os.path.join(root, run_name, "sft", "final"),
        "dpo": os.path.join(root, run_name, "dpo", "final"),
    }
