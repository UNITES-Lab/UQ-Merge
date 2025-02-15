import itertools
import json
import os
import re

import rich
from natsort import humansorted, os_sorted

root_dir = "/home/xxx/workspace/mergekit/average_merging/input_perturbation_asce"

result_dirs = os_sorted([d.path for d in os.scandir(root_dir) if d.is_dir()])[1:]
# recipes = [f"{root_dir}/{d}" for d in os.listdir(root_dir) if not d.endswith(".yml")]

for i, result_dir in enumerate(result_dirs):
    eval_dir = os_sorted([d.path for d in os.scandir(result_dir) if d.is_dir()])[-1]
    print(eval_dir)
    result_dirs[i] = eval_dir 

average_accs = []
for i, result_dir in enumerate(result_dirs):
    accs = []
    with open(f"{result_dir}/results.json") as f:
        result_json = json.load(f)
    for dataset in result_json["results"].keys():
        accs.append(list(result_json["results"][dataset].values())[0])
    average_accs.append(sum(accs) / len(accs))

    rich.print(len(result_dir.split("/")[-2].split("-")) - 1, sum(accs) / len(accs) * 100)
