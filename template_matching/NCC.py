import cv2
import numpy as np

img = cv2.resize(
    cv2.imread(r"C:\Users\KIRUBA\Documents\Auto-Annotation\template_matching\images\birds in the sky.jpg"), 
    None, 
    fx=0.4, 
    fy=0.4
    )
img = cv2.GaussianBlur(img, (5, 5), 0.3)
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # ret, thresotsu = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# img = cv2.Canny(gray_img, 0, 255)

drawing = False
ix, iy = -1, -1
top_left, bottom_right = None, None
template = None

def draw_rect(event, x, y, flag, param):
    
    global drawing, ix, iy, top_left, bottom_right, template
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = img.copy()
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow("Auto Annotate", img_copy)
            
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        top_left = ix, iy
        bottom_right = x, y
        if top_left[1] < bottom_right[1]:
            template = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        else:
            template = img[bottom_right[1]:top_left[1], bottom_right[0]:top_left[0]]
        cv2.rectangle(img, top_left, bottom_right, (255, 0, 0), 2)
        if template is not None:
            template_matching()
            
def template_matching():
    threshold = 0.4
    tem_h, tem_w = template.shape[:2]
    h, w = img.shape[:2]
    
    tem_mean = np.mean(template)
    tem_norm = template - tem_mean
    
    norm_cc = np.zeros((h - tem_h + 1, w - tem_w + 1))
    
    bbox = []
    scores = []
    for y in range(norm_cc.shape[0]):
        for x in range(norm_cc.shape[1]):
            roi = img[y: y+tem_h, x: x+tem_w]
            
            roi_mean = np.mean(roi)
            roi_norm = roi - roi_mean
            
            upper = np.sum(roi_norm * tem_norm)
            lower = np.sqrt(np.sum(roi_norm ** 2) * np.sum(tem_norm ** 2))
            
            if lower != 0:
                score = upper / lower
                norm_cc[y, x] = score
                if score >= threshold:
                    bbox.append((x, y, x+tem_w, y+tem_h))
                    scores.append(score)
                    
    
    if len(bbox) == 0:
        return []
    
    bboxes = np.array(bbox)
    scores = np.array(scores)
    
    area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    sort_socres = np.argsort(scores)[::-1]

    keep = []
    while len(sort_socres) > 0:
        current_ind = sort_socres[0]
        keep.append(current_ind)
        current_box = bboxes[current_ind]
        remaining_boxes = bboxes[sort_socres[1:]]
        
        x1 = np.maximum(current_box[0], remaining_boxes[:, 0]) 
        y1 = np.maximum(current_box[1], remaining_boxes[:, 1])
        x2 = np.minimum(current_box[2], remaining_boxes[:, 2])
        y2 = np.minimum(current_box[3], remaining_boxes[:, 3])
        
        inter_width = np.maximum(0, (x2 - x1)) 
        inter_height = np.maximum(0, (y2-y1)) 
        
        intersection = inter_width * inter_height
        
        union = area[current_ind] + area[sort_socres[1:]] - intersection
        
        iou = intersection / union 
        
        filtered_indexes = np.where(iou <= 0.5)[0]
        sort_socres = sort_socres[filtered_indexes + 1]
    count = 1 
    for box in keep:
        bounding_box = bbox[box]
        x, y, xw, yh = bounding_box
        cv2.rectangle(img, (x, y), (xw, yh), (255, 0, 0), 2)
        cv2.putText(img, f"{count}", (x+2, y+13), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
        print(f"Imgae no {count}, BBOX: {(x, y, xw, yh)}")
        count += 1
    cv2.imshow("Auto Annotate", img)
    # print(f"scores{scores}")
    # print(f"boxes{bboxes}")
    # print(f"areas : {area}")
    # print(f"sort_index : {sort_socres}")
    
cv2.namedWindow("Auto Annotate")
cv2.setMouseCallback("Auto Annotate", draw_rect)
cv2.imshow("Auto Annotate", img)
cv2.waitKey(0)
cv2.destroyAllWindows()