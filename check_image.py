from PIL import Image, ImageDraw
import numpy as np

img_path = "C:/Users/Tommy/Downloads/SegmentationObject/D21 [x=67072,y=18432,w=512,h=512].png"

img_pil = Image.open(img_path)
img_np = np.array(img_pil)

img_unique = np.unique(img_np)

print()
