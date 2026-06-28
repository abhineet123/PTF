import os

# import matplotlib.pyplot as plt
import numpy as np

# import sam3
from PIL import Image
from sam3 import build_sam3_image_model

# from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor

# from sam3.visualization_utils import draw_box_on_image, normalize_bbox, plot_results


import torch


def copy_from_clipboard():
    try:
        import win32clipboard

        win32clipboard.OpenClipboard()
        in_txt = win32clipboard.GetClipboardData()
    except BaseException as e:
        print("GetClipboardData failed: {}".format(e))
        win32clipboard.CloseClipboard()
        return None
    win32clipboard.CloseClipboard()
    return in_txt


# turn on tfloat32 for Ampere GPUs
# https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# torch.autocast("cuda", dtype=torch.bfloat16).__enter__()


def main():

    model = build_sam3_image_model()

    # sam3_root = os.path.join(os.path.expanduser("~"), "sam3")
    # image_path = f"{sam3_root}/assets/images/AH_000191.jpg"
    image_path = copy_from_clipboard()

    image = Image.open(image_path)
    processor = Sam3Processor(model, confidence_threshold=0.5)
    inference_state = processor.set_image(image)

    processor.reset_all_prompts(inference_state)
    output = processor.set_text_prompt(state=inference_state, prompt="person")

    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
    masks = masks.detach().cpu().numpy().squeeze()

    # boxes = boxes.detach().cpu().numpy().squeeze()
    # scores = scores.detach().cpu().numpy().squeeze()

    image_np = np.array(image)

    for mask_id, mask in enumerate(masks):
        mask_uint8 = mask.astype(np.uint8) * 255
        masked_image = np.zeros_like(image_np)
        masked_image[mask] = image_np[mask]
        Image.fromarray(mask_uint8).save(f"mask_{mask_id}.png")
        Image.fromarray(masked_image).save(f"masked_image_{mask_id}.png")


if __name__ == "__main__":
    main()
