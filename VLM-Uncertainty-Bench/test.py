from PIL import Image
from torchvision.transforms import v2 as T

augmenter = T.AugMix()
image_path = '/home/xxx/workspace/VLM-Uncertainty-Bench/3.png'
image = Image.open(image_path).convert('RGB')
image_tensor = T.functional.pil_to_tensor(image).cuda()
perturbated_tensor = augmenter(image_tensor)
image = T.functional.to_pil_image(perturbated_tensor)
image.save("perturbed_3.jpg")
