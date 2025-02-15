import pickle
import numpy as np
import os

def entropy(probs):
    return -np.sum(probs * np.log(probs + 1e-10), axis=-1)

root_dir = "/home/xxx/workspace/VLM-Uncertainty-Bench/merging_steps/llava-gqa-okvqa-a_okvqa-llava_instruct-ocrvqa-vg-vqav2-sharegpt-refcoco-textcaps"

average = []
for dataset in os.listdir(root_dir):
    with open(f"{root_dir}/{dataset}", "rb") as f:
        results = pickle.load(f)

    logits = np.array([answer["logits"] for answer in results])[:, :-1]
    # print(logits)
    probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)

    # print(probs[:4])

    epistemic = entropy(probs.reshape(-1, 4, 6).mean(axis=1))
    aleatoric = entropy(probs.reshape(-1, 4, 6)).mean(axis=1)

    uncertainty = epistemic - aleatoric
    print(dataset)
    print(uncertainty[~np.isnan(uncertainty)].mean())
    average.append(uncertainty[~np.isnan(uncertainty)].mean())

print("\n", sum(average) / len(average))
    