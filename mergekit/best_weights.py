import itertools
import json
import os
import re

import rich
from natsort import humansorted, os_sorted

root_dir = "/xxx/xxx/mergekit/ckpts/UQ_relation"
recipes = [f"{root_dir}/{d}" for d in os.listdir(root_dir) if not d.endswith(".yml")]

result_dirs = []
for recipe in recipes:
    result_dir = os_sorted([d.path for d in os.scandir(recipe) if d.is_dir()])[-1]
    result_dirs.append(result_dir)

for i, result_dir in enumerate(result_dirs):
    if "results.json" not in os.listdir(result_dir):
        print(result_dir)
        result_dirs.pop(i)

average_accs = []
for i, result_dir in enumerate(result_dirs):
    accs = []
    with open(f"{result_dir}/results.json") as f:
        result_json = json.load(f)
    for dataset in result_json["results"].keys():
        accs.append(list(result_json["results"][dataset].values())[0])
    average_accs.append(sum(accs) / len(accs))

datasets = ["okvqa", "ocrvqa", "gqa", "vqav2"]

results = {}
for m1, m2 in itertools.combinations(datasets, 2):
    results[f"{m1}-{m2}"] = {}
    for i, result_dir in enumerate(result_dirs):
        name = result_dir.split("/")[-2]
        if m1 in name and m2 in name:
            w1, w2 = re.findall(r'\d+\.\d+', name)
            results[f"{m1}-{m2}"][f"{w1}-{w2}"] = average_accs[i]

rich.print(results)