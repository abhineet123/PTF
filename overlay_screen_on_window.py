import cv2
import numpy as np

from Misc import resize_ar, linux_path

root_dir = "X:/UofA/neon_vista/screen/EU"

window_file = "eu_window_person.webp"
screen_file = "screen.png"
mask_file = "eu_window_person_mask.png"
screen_size = 65

window_x, window_y = 130, 188
window_h, window_w = 498, 382

camera_height = 390
screen_size_to_hw = {
    32: (153, 91),
    43: (205, 120),
    49: (233, 137),
    55: (262, 154),
    65: (310, 181),
}
screen_h, screen_w = screen_size_to_hw[screen_size]

window_img = cv2.imread(linux_path(root_dir, window_file))
screen_img = cv2.imread(linux_path(root_dir, screen_file))
mask_img = cv2.imread(linux_path(root_dir, mask_file)).astype(bool)

screen_img_res = cv2.resize(
    screen_img, (screen_w, screen_h),
    # interpolation=cv2.INTER_LANCZOS4,
    interpolation=cv2.INTER_AREA,
)

# screen_img_res = resize_ar(
#     screen_img,
#     # width=screen_w,
#     height=screen_h,
# )

window_img_h, window_img_w = window_img.shape[:2]
screen_img_h, screen_img_w = screen_img_res.shape[:2]

border_x = int((window_w - screen_w) / 2)

screen_x, screen_y = window_x + border_x, window_y + window_h - camera_height

border_y1 = screen_y - window_y
border_y2 = (window_y + window_h) - (screen_y + screen_img_w)

print(f"border_x: {border_x}")
print(f"border_y1: {border_y1}")
print(f"border_y2: {border_y2}")

overlaid_img = np.copy(window_img)
overlaid_img[screen_y:screen_y + screen_img_h, screen_x:screen_x + screen_img_w, ...] = screen_img_res
overlaid_img[mask_img] = window_img[mask_img]

cv2.imshow("overlaid_img", overlaid_img)
cv2.waitKey(0)

cv2.imwrite(linux_path(root_dir, f"overlaid_{screen_size}_inches.png"), overlaid_img)
