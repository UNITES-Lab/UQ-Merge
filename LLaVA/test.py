import json
import os

with open("/xxx/xxx/LLaVA/playground/llava_v1_5_mix665k.json", "r") as f:
    s = json.load(f)

aokvqa_samples = []
for item in s:
    try:
        if "ocr_vqa" in item["image"]:
            aokvqa_samples.append(item)
    except:
        pass
print(len(aokvqa_samples))

json_string = json.dumps(aokvqa_samples, indent=4)
with open("/xxx/xxx/LLaVA/playground/ocrvqa.json", "w") as f:
    f.write(json_string)