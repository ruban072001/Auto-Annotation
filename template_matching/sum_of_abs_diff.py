import cv2
import numpy as np

"""
Template matching based on sum of squared difference
original_image size = 1089, 1500, 3
template_size = 60, 137, 3
"""

Original_img = cv2.imread(r"C:\Users\KIRUBA\Pictures\birds in the sky.jpg")
Original_img = cv2.resize(Original_img, (0,0), fx=0.3, fy=0.3)
template = cv2.imread(r"C:\Users\KIRUBA\Downloads\birds in the sky (1).jpg")
template = cv2.resize(template, (0,0), fx=0.3, fy=0.3)

ori_img_h, ori_img_w = Original_img.shape[:2]
print(Original_img.shape)
tem_h, tem_w = template.shape[:2]
print(template.shape)

SAD_map = np.zeros((ori_img_h - tem_h, ori_img_w - tem_w))

for i in range(ori_img_h - tem_h):
    for j in range(ori_img_w - tem_w):
        
        roi = Original_img[i : i + tem_h, j : j + tem_w]
        ssd = np.sum(np.abs(roi - template))
        
        SAD_map[i, j] = ssd
        
# print(SSD_map)
print(SAD_map.shape)
min_val = np.min(SAD_map) ; print(min_val)
# min_loc = np.unravel_index(np.argmin(SSD_map), SSD_map.shape) ; print(min_loc)
threshold = 155000
matches = np.where(SAD_map <= threshold)
# print(matches)
for match in zip(matches[0], matches[1]):
    top_left = match[1], match[0]
    bottom_right = top_left[0] + tem_w, top_left[1] + tem_h

    cv2.rectangle(Original_img, top_left, bottom_right, (0, 255, 0), 2)
cv2.imshow('template', Original_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
        
