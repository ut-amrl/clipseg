import torch

from models.clipseg import CLIPDensePredT
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np

# load model
# model = CLIPDensePredT(version="ViT-B/16", reduce_dim=64)
# model.eval()

# non-strict, because we only stored decoder weights (not CLIP weights)
# model.load_state_dict(
#     torch.load("weights/rd64-uni.pth", map_location=torch.device("cpu")),
#     strict=False,
# )

model = CLIPDensePredT(version="ViT-B/16", reduce_dim=64, complex_trans_conv=True)
model.load_state_dict(
    torch.load("weights/rd64-uni-refined.pth", map_location=torch.device("cpu")),
    strict=False,
)

# load and normalize image
input_image = Image.open("assets/campus1.jpeg")
prompts = [
    "people",
    "lampposts",
    "bricks",
]

# input_image = Image.open("assets/campus2.jpg")
# prompts = [
#     "people",
#     "lampposts",
#     "bricks",
# ]

# input_image = Image.open("assets/campus3.jpeg")
# prompts = [
#     "a building",
#     "a sign with the word 'corssroads'",
#     "people",
# ]

# input_image = Image.open("assets/campus3.jpeg")
# prompts = [
#     "a building",
#     "a red building with white roof",
#     "people",
#     "a person with orange jacket",
# ]

# input_image = Image.open("assets/urban1.jpeg")
# prompts = ["cars", "people", "bikes"]

# input_image = Image.open("assets/urban2.jpeg")
# prompts = ["cars", "people", "bikes"]

# input_image = Image.open("assets/urban2.jpeg")
# prompts = ["cars", "people", "bikes"]

# input_image = Image.open("assets/urban3.jpeg")
# prompts = ["buses", "traffic lights", "lampposts", "a car"]

# input_image = Image.open("assets/dog1.jpeg")
# prompts = ["a black dog", "a white dog", "dog on the right", "dog on the left"]

# input_image = Image.open("assets/hands1.jpeg")
# prompts = ["left hand", "right hand"]

# input_image = Image.open("assets/coda1.png")
# prompts = [
#     "cars",
#     "occluded cars",
# ]

# input_image = Image.open("assets/coda1.png")
# prompts = [
#     "people",
#     "a person with yellow shirt",
#     "people wearing jeans",
#     "the person closest to camera",
# ]

tdim = 512

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        # TODO replace the hardcoding with calculated normalization
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize((tdim, tdim)),
    ]
)
img = transform(input_image).unsqueeze(0)

n_prompts = len(prompts)

with torch.no_grad():
    preds = model(img.repeat(n_prompts, 1, 1, 1), prompts)[0]

# visualize prediction
_, ax = plt.subplots(1, n_prompts + 1, figsize=(15, 4))
[a.axis("off") for a in ax.flatten()]
ax[0].imshow(input_image.resize((tdim, tdim)))
[ax[i + 1].imshow(torch.sigmoid(preds[i][0])) for i in range(n_prompts)]
[ax[i + 1].text(0, -15, prompts[i]) for i in range(n_prompts)]
plt.show()
