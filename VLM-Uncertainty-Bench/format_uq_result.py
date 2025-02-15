import glob
import json
import os

import rich

root = "llava_single"

def find_json_files(root_dir):
    json_files = glob.glob(os.path.join(root_dir, '**/*.json'), recursive=True)
    return json_files

uq_results = find_json_files(root)

for uq_result in uq_results:
    name = uq_result.split("/")[1]

    with open(uq_result) as f:
        result = json.load(f)

    new_format = {name: {}}
    for dataset, value in result.items():
        new_format[name][dataset] = value["acc"][0] * 100
        # new_format.append(value["set_sizes"][0])
        # new_format.append(value["uacc"][0] * 100)
        # new_format.append(value["ece"][0])
        # new_format.append(value["mce"][0])

    # print(name.count("+") + 1)
    # print(name)
    # print(new_format)
    # print(sum(new_format[0::2]) / 5)
    # print(sum(new_format[1::2]) / 5)
    # print()

    rich.print(new_format)
