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

template_mean = np.mean(template)
template_normalize = template - template_mean

ncc_map = np.zeros((ori_img_h - tem_h + 1, ori_img_w - tem_w + 1))
for i in range(ori_img_h - tem_h + 1):
    for j in range(ori_img_w - tem_w + 1):
        
        roi = Original_img[i : i + tem_h, j : j + tem_w]
        roi_mean = np.mean(roi)
        
        roi_normalization = roi - roi_mean
        
        numenator = np.sum(roi_normalization * template_normalize)
        denominator = np.sqrt(np.sum(roi_normalization ** 2) * np.sum(template_normalize ** 2))
        
        if denominator != 0:
            ncc_map[i, j] = numenator / denominator
        else:
            ncc_map[i, j] = 0
        
# print(SSD_map)
print(ncc_map.shape)
max_val = np.max(ncc_map) ; print(max_val)
# min_loc = np.unravel_index(np.argmin(SSD_map), SSD_map.shape) ; print(min_loc)
threshold = 0.6
matches = np.where(ncc_map >= threshold)
# print(matches)
for match in zip(matches[0], matches[1]):
    top_left = match[1], match[0]
    bottom_right = top_left[0] + tem_w, top_left[1] + tem_h
    
    cv2.rectangle(Original_img, top_left, bottom_right, (0, 255, 0), 2)
cv2.imshow('template', Original_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
        
