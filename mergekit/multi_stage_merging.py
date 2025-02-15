import itertools
import json
import os
import re
import random

import rich
import yaml
from natsort import humansorted, os_sorted

model_base_root = "LLaVA/new_checkpoints/llava-v1.5-7b"

# datasets = ["okvqa", "ocrvqa", "gqa", "vqav2", "textcaps", "llava_instruct", "sharegpt", "vg", "a_okvqa", "refcoco"] # 0order, random
# datasets = ["gqa", "vqav2", "ocrvqa", "okvqa", "vg", "llava_instruct", "sharegpt", "a_okvqa", "refcoco", "textcaps"] # descending test acc, the second one
# random.shuffle(datasets)

# datasets = ["okvqa", "ocrvqa", "gqa", "vqav2", "a_okvqa", "refcoco", "llava_instruct", "vg"]
# random.shuffle(datasets)

# datasets = ["okvqa", "a_okvqa", "vqav2", "refcoco", "sharegpt", "llava_instruct", "vg", "ocrvqa", "textcaps", "gqa"]
# datasets = ["okvqa", "a_okvqa", "vqav2", "refcoco", "sharegpt", "llava_instruct", "vg", "ocrvqa"]
# random.shuffle(datasets)

# datasets = ["gqa", "vqav2", "ocrvqa", "okvqa", "vg", "llava_instruct", "a_okvqa", "refcoco", ]
# datasets = ["okvqa", "a_okvqa", "vqav2", "refcoco", "llava_instruct", "vg", "ocrvqa", "gqa"]
# datasets = ["vqav2", "ocrvqa", "okvqa", "vg", "llava_instruct", "sharegpt", "a_okvqa", "refcoco"]
# datasets = ["ocrvqa", "vqav2", "vg", "sharegpt", "refcoco", "okvqa", "gqa", "a_okvqa", "llava_instruct", "textcaps"]
# datasets = ["ocrvqa", "vqav2", "vg", "sharegpt", "refcoco", "okvqa", "gqa", "a_okvqa"]
# datasets = ["gqa", "vqav2", "ocrvqa", "okvqa", "vg", "sharegpt", "a_okvqa", "refcoco"]
# datasets = ["gqa", "vqav2", "ocrvqa", "okvqa", "vg", "llava_instruct", "sharegpt", "a_okvqa"]
# datasets = ["okvqa", "a_okvqa", "vqav2", "sharegpt", "llava_instruct", "vg", "ocrvqa", "gqa"]
# datasets = ["ocrvqa", "vqav2", "vg", "sharegpt", "okvqa", "gqa", "a_okvqa", "llava_instruct"]

# datasets = ["gqa", "vqav2", "ocrvqa", "okvqa", "vg", "llava_instruct", "sharegpt", "a_okvqa", "refcoco", "textcaps"] # descending test acc, the second one
# random.shuffle(datasets)
# datasets = ["textcaps", "vg", "sharegpt", "refcoco", "vqav2", "llava_instruct", "ocrvqa", "okvqa", "gqa", "a_okvqa"]
# datasets = ["textcaps", "vg", "sharegpt", "refcoco", "vqav2", "llava_instruct", "okvqa", "ocrvqa", "gqa", "a_okvqa"]
# datasets = ["a_okvqa", "gqa", "okvqa", "ocrvqa", "llava_instruct", "vqav2", "refcoco", "sharegpt", "vg", "textcaps"]
# random.shuffle(datasets)

datasets = ["gqa", "okvqa", "a_okvqa", "llava_instruct", "ocrvqa", "vg", "vqav2", "sharegpt", "refcoco", "textcaps"]

merged = "llava"
for i, dataset in enumerate(datasets):
    if i == 0:
        merged = merged + f"-{dataset}"
        os.makedirs(f"mergekit/dynamic_merging/perturbation_model1/{merged}", exist_ok=True)
    elif i == 1:
        merged = merged + f"-{dataset}"

        recipe = {
            "models": [
                {
                    "model": f"{model_base_root}-{datasets[0]}",
                    "parameters": {"weight": 1}
                },
                {
                    "model": f"{model_base_root}-{datasets[1]}",
                    "parameters": {"weight": 1}
                }
            ],
            "merge_method": "linear",
            "dtype": "float16"
        }
        print(recipe)
        config = f'mergekit/dynamic_merging/perturbation_model1/{merged}.yml'
        with open(config, 'w') as file:
            yaml.dump(recipe, file, default_flow_style=False, sort_keys=False)

        os.system(f"mergekit-yaml {config} mergekit/dynamic_merging/perturbation_model1/{merged}")
        prev_model = f"mergekit/dynamic_merging/perturbation_model1/{merged}"
    else:
        merged = merged + f"-{dataset}"

        recipe = {
            "models": [
                {
                    "model": prev_model,
                    "parameters": {"weight": 1}
                },
                {
                    "model": f"{model_base_root}-{datasets[i]}",
                    "parameters": {"weight": 1}
                }
            ],
            "merge_method": "linear",
            "dtype": "float16"
        }
        print(recipe)
        config = f'mergekit/dynamic_merging/perturbation_model1/{merged}.yml'
        with open(config, 'w') as file:
            yaml.dump(recipe, file, default_flow_style=False, sort_keys=False)

        os.system(f"mergekit-yaml {config} mergekit/dynamic_merging/perturbation_model1/{merged}")
        prev_model = f"mergekit/dynamic_merging/perturbation_model1/{merged}"
