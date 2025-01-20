import cv2
import numpy as np

Original_img = cv2.imread(r"C:\Users\KIRUBA\Pictures\birds in the sky.jpg")
img = cv2.resize(Original_img, (0,0), fx=0.3, fy=0.3)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print(f"gray img : {gray_img}")
h, w = gray_img.shape[:2]

kernel_x = np.array([[-1, 0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]], dtype=np.float32)

kernel_y = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]])

gradient_x = np.zeros((h, w), dtype=np.float32)
gradient_y = np.zeros((h, w), dtype=np.float32)
padded_img = np.pad(gray_img, pad_width=1, mode='constant', constant_values=0)
# print(f"padded img : {padded_img}")
for i in range(1, h + 1):
    for j in range(1, w + 1):
        roi = padded_img[i-1:i+2, j-1:j+2]
        
        grad_x = kernel_x * roi
        
        gradient_x[i-1, j-1] = np.sum(grad_x)
        
        
for i in range(1, h+1):
    for j in range(1, w+1):
        roi = padded_img[i-1:i+2, j-1:j+2]
        
        grad_y = kernel_y * roi
        
        gradient_y[i-1, j-1] = np.sum(grad_y)
# print(f"gradient:  {gradient_x}")
result_x = cv2.convertScaleAbs(gradient_x)
result_y = cv2.convertScaleAbs(gradient_y)
# print(f"result:  {result_x}")


# cv2.imshow("gray img", gray_img)
# cv2.imshow("gradient_x", gradient_x)
# cv2.imshow("gradient_y", gradient_y)
# cv2.imshow("result_x", result_x)
# cv2.imshow("result_y", result_y)
# cv2.waitKey(0)
# cv2.destroyAllWindows()