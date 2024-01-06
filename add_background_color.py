from PIL import Image
import numpy as np
import cv2

im = Image.open(r"C:\Users\Tommy\Pictures\Wallpapers\Pics\#\manipulation-artist-graphy-others-png-clip-art.png")

fill_color = (0, 255, 0)  # your new background color

im = im.convert("RGBA")  # it had mode P after DL it from OP
if im.mode in ('RGBA', 'LA'):
    background = Image.new(im.mode[:-1], im.size, fill_color)
    background_cv = np.copy(np.array(background))
    background.paste(im, im.split()[-1])  # omit transparency
    background_cv2 = np.copy(np.array(background))

    im = background

im_rgb = im.convert("RGB")
im_rgb.save(r"C:\Users\Tommy\Pictures\Wallpapers\Pics\#\manipulation-artist-graphy-others-png-clip-art2.png")
im_rgb_cv = np.array(im_rgb)

cv2.imshow('background_cv', background_cv)
cv2.imshow('background_cv2', background_cv2)
cv2.imshow('im_rgb_cv', im_rgb_cv)

cv2.waitKey(0)



