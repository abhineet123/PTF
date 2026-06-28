import os

import matplotlib.pyplot as plt
import numpy as np

import sam3
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import draw_box_on_image, normalize_bbox, plot_results

sam3_root = os.path.join(os.path.expanduser("~"), "sam3")

import torch

# turn on tfloat32 for Ampere GPUs
# https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# use bfloat16 for the entire notebook
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

model = build_sam3_image_model()

image_path = f"{sam3_root}/assets/images/AH_000191.jpg"
image = Image.open(image_path)
width, height = image.size
processor = Sam3Processor(model, confidence_threshold=0.5)
inference_state = processor.set_image(image)

processor.reset_all_prompts(inference_state)
output = processor.set_text_prompt(state=inference_state, prompt="person")

masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
mask = masks.detach().cpu().numpy().squeeze()
Image.fromarray(mask).save(f"mask.png")












