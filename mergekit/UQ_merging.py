import itertools
import os

import yaml

model_base_root = "/xxx/xxx/LLaVA/new_checkpoints/llava-v1.5-7b"
datasets = ["okvqa", "ocrvqa", "gqa", "vqav2"]

for m1, m2 in itertools.combinations(datasets, 2):
    for w in range(1, 10):
        recipe = {
            "models": [],
            "merge_method": "linear",
            "dtype": "float16"
        }

        c1 = round(w / 10, 1)
        c2 = round(1 - w / 10, 1)

        x = {
            "model": f"{model_base_root}-{m1}",
            "parameters": {
                "weight": c1
            }
        }
        y = {
            "model": f"{model_base_root}-{m2}",
            "parameters": {
                "weight": c2
            }
        }

        recipe["models"].append(x)
        recipe["models"].append(y)

        config = f'ckpts/UQ_relation/{m1}_{c1}-{m2}_{c2}.yml'
        with open(config, 'w') as file:
            yaml.dump(recipe, file, default_flow_style=False, sort_keys=False)

        os.system(f"mergekit-yaml {config} ckpts/UQ_relation/{m1}_{c1}-{m2}_{c2}")
