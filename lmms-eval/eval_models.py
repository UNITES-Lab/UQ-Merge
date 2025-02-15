# import os
# import re
# import subprocess

# # tasks = "gqa_lite,vizwiz_vqa_val_lite,vqav2_val_lite,textvqa_val_lite,ok_vqa_val2014_lite"mmbench_en_dev_lite

# root_dir = "/xxx/xxx/mergekit/multi_stage_merging/order0"

# models = [m for m in os.listdir(root_dir) if not m.endswith("yml")]

# # weights = []
# # for model in models:
# #     w1, w2 = re.findall(r'\d+\.\d+', model)
# #     weights.append((float(w1), float(w2)))

# for i, model in enumerate(models):
#     if len(model.split("-")) == 2:
#         continue
#     command = \
#     f"""
#     MODEL_PATH="{root_dir}/{model}" MODEL_NAME="{model}" \
#     OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0,1,2,3 \
#     python3 -m accelerate.commands.launch --num_processes=4 --main_process_port 12345 -m lmms_eval --model llava \
#     --model_args pretrained="liuhaotian/llava-v1.5-7b" \
#     --tasks gqa_lite,vizwiz_vqa_val_lite,vqav2_val_lite,textvqa_val_lite,ok_vqa_val2014_lite,ai2d_lite,docvqa_val_lite \
#     --batch_size 1 --log_samples --log_samples_suffix {model} \
#     --output_path "{root_dir}/{model}"
#     """
#     os.system(command)

import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

for order in ["model_perturbation-asec"]:
    root_dir = f"/home/xxx/workspace/mergekit/average_merging/{order}"
    models = [m for m in os.listdir(root_dir) if not m.endswith("yml")]

    def run_command(model, gpu_id):
        if len(model.split("-")) == 2:
            return

        command = f"""
        MODEL_PATH="{root_dir}/{model}" MODEL_NAME="{model}" \
        OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES={gpu_id} \
        python3 -m accelerate.commands.launch --num_processes=1 --main_process_port 12345 -m lmms_eval --model llava \
        --model_args pretrained="liuhaotian/llava-v1.5-7b" \
        --tasks realworldqa,nextqa_mc_test,ocrbench,doc_vqa_lite \
        --batch_size 1 --log_samples --log_samples_suffix {model} \
        --output_path "{root_dir}/{model}"
        """
        os.system(command)

    def worker(gpu_id, task_queue):
        while not task_queue.empty():
            model = task_queue.get()
            run_command(model, gpu_id)
            task_queue.task_done()

    # queue shared across threads
    task_queue = Queue()

    # add all models to the queue
    for model in models:
        task_queue.put(model)

    # create threads for tasks, one thread on each gpu
    with ThreadPoolExecutor(max_workers=4) as executor:
        for gpu_id in [5, 6, 7]:
            executor.submit(worker, gpu_id, task_queue)

    # wait
    task_queue.join()
