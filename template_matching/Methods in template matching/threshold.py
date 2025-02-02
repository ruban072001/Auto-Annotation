import cv2
import numpy as np

img = cv2.resize(
    cv2.imread(r"C:\Users\KIRUBA\Documents\Auto-Annotation\template_matching\images\birds in the sky.jpg"),
    None,
    fx=0.4,
    fy=0.4,
)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresotsu = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
canny = cv2.Canny(gray_img, 0, 255)



cv2.imshow("otsu", thresotsu)
cv2.imshow("bin", canny)
cv2.waitKey(0)
cv2.destroyAllWindows()