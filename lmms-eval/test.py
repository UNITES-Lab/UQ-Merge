context = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\nWhat type of ant is Ant D?\nA. Fire Ant\nB. Pharoah Ant\nC. Crazy Ant\nD. Carpenter Ant\nAnswer with the option's letter from the given choices directly. ASSISTANT:"

sentences = context.split('\n')
print(sentences)

sentences[-1] = "Answer your confidence about the answer using a float score between 0.0 to 1.0. ASSISTANT:"

sentences = "\n".join(sentences)
print(sentences)
