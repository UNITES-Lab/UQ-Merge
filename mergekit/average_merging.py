import itertools
import json
import os
import random
import re

import rich
import yaml
from natsort import humansorted, os_sorted

model_base_root = "/home/xxx/workspace/LLaVA/new_checkpoints/llava-v1.5-7b"

datasets = ["gqa", "okvqa", "a_okvqa", "llava_instruct", "ocrvqa", "vg", "vqav2", "sharegpt", "refcoco", "textcaps"]
# datasets = ["textcaps", "refcoco", "sharegpt", "vqav2", "vg", "ocrvqa", "llava_instruct", "a_okvqa", "okvqa", "gqa"]
# random.shuffle(datasets)

# datasets = ["textcaps", "refcoco", "sharegpt", "vqav2", "vg", "ocrvqa", "llava_instruct", "a_okvqa", "okvqa", "gqa"]
# datasets = ["a_okvqa", "ocrvqa", "refcoco", "llava_instruct", "gqa", "vg", "sharegpt", "vqav2", "okvqa", "textcaps"]
# datasets = ["textcaps", "okvqa", "vqav2", "sharegpt", "vg", "gqa", "llava_instruct", "refcoco", "ocrvqa", "a_okvqa"]

# datasets = ["a_okvqa", "gqa", "okvqa", "ocrvqa", "llava_instruct", "vqav2", "refcoco", "sharegpt", "vg", "textcaps"]
# datasets = ["textcaps", "vg", "sharegpt", "refcoco", "vqav2", "llava_instruct", "ocrvqa", "okvqa", "gqa", "a_okvqa"]

recipe = {
    "models": [],
    "merge_method": "della",
    "base_model": "/home/xxx/workspace/LLaVA/new_checkpoints/llava-v1.5-7b-raw",
    "dtype": "float16"
}

merged = "llava"
for i, dataset in enumerate(datasets):
    if i == 0:
        merged = merged + f"-{dataset}"
        os.makedirs(f"/home/xxx/workspace/mergekit/average_merging/della/{merged}", exist_ok=True)
        recipe["models"].append({"model": f"{model_base_root}-{dataset}", "parameters": {"weight": 1}})
    elif i == 1:
        merged = merged + f"-{dataset}"
        recipe["models"].append({"model": f"{model_base_root}-{dataset}", "parameters": {"weight": 1}})
        print(recipe)
        config = f'/home/xxx/workspace/mergekit/average_merging/della/{merged}.yml'
        with open(config, 'w') as file:
            yaml.dump(recipe, file, default_flow_style=False, sort_keys=False)

        os.system(f"mergekit-yaml {config} /home/xxx/workspace/mergekit/average_merging/della/{merged}")
        prev_model = f"/home/xxx/workspace/mergekit/average_merging/della/{merged}"
    # elif i == 6:
    #     merged = merged + f"-{dataset}"
    #     recipe["models"].append({"model": f"{model_base_root}-{dataset}", "parameters": {"weight": 1}})
    #     print(recipe)
    #     config = f'/home/xxx/workspace/mergekit/average_merging/della/{merged}.yml'
    #     with open(config, 'w') as file:
    #         yaml.dump(recipe, file, default_flow_style=False, sort_keys=False)

    #     os.system(f"mergekit-yaml {config} /home/xxx/workspace/mergekit/average_merging/della/{merged}")
    #     prev_model = f"/home/xxx/workspace/mergekit/average_merging/della/{merged}"
    else:
        merged = merged + f"-{dataset}"
        recipe["models"].append({"model": f"{model_base_root}-{dataset}", "parameters": {"weight": 1}})
        print(recipe)
        config = f'/home/xxx/workspace/mergekit/average_merging/della/{merged}.yml'
        with open(config, 'w') as file:
            yaml.dump(recipe, file, default_flow_style=False, sort_keys=False)

        os.system(f"mergekit-yaml {config} /home/xxx/workspace/mergekit/average_merging/della/{merged}")
        prev_model = f"/home/xxx/workspace/mergekit/average_merging/della/{merged}"
