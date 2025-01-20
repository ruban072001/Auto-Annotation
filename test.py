import cv2
import numpy as np

# Step 1: Harris Corner Detection for Keypoints
def harris_corner_detector(image, k=0.04, threshold=0.01):
    gray = np.float32(image)
    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # Gradient in X
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # Gradient in Y
    Ixx = Ix ** 2
    Iyy = Iy ** 2
    Ixy = Ix * Iy

    # Compute response R for each pixel
    height, width = gray.shape
    R = np.zeros_like(gray)
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            # 3x3 window
            Sxx = Ixx[y - 1:y + 2, x - 1:x + 2].sum()
            Syy = Iyy[y - 1:y + 2, x - 1:x + 2].sum()
            Sxy = Ixy[y - 1:y + 2, x - 1:x + 2].sum()

            # Compute Harris response
            det = (Sxx * Syy) - (Sxy ** 2)
            trace = Sxx + Syy
            R[y, x] = det - k * (trace ** 2)

    # Normalize and threshold R
    R = cv2.normalize(R, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    keypoints = np.argwhere(R > threshold)

    return keypoints

# Step 2: Generate ORB Descriptors
def generate_orb_descriptors(image, keypoints, patch_size=31):
    descriptors = []
    for y, x in keypoints:
        # Extract a patch around the keypoint
        if y < patch_size // 2 or x < patch_size // 2 or \
           y >= image.shape[0] - patch_size // 2 or \
           x >= image.shape[1] - patch_size // 2:
            continue  # Skip keypoints too close to the edge
        patch = image[y - patch_size // 2:y + patch_size // 2 + 1,
                      x - patch_size // 2:x + patch_size // 2 + 1]

        # Compute intensity comparisons to generate a binary descriptor
        center = patch[patch_size // 2, patch_size // 2]
        descriptor = []
        for dy in range(-patch_size // 2, patch_size // 2 + 1):
            for dx in range(-patch_size // 2, patch_size // 2 + 1):
                if dy == 0 and dx == 0:
                    continue
                descriptor.append(1 if patch[dy + patch_size // 2, dx + patch_size // 2] > center else 0)
        descriptors.append((x, y, np.array(descriptor)))

    return descriptors

# Step 3: Match Descriptors
def match_descriptors(descriptors1, descriptors2):
    matches = []
    for (x1, y1, desc1) in descriptors1:
        best_distance = float('inf')
        best_match = None
        for (x2, y2, desc2) in descriptors2:
            distance = np.sum(desc1 != desc2)  # Hamming distance
            if distance < best_distance:
                best_distance = distance
                best_match = (x1, y1, x2, y2)
        matches.append(best_match)
    return matches

# Main Function
def feature_based_matching(image1, image2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Detect keypoints
    keypoints1 = harris_corner_detector(gray1)
    keypoints2 = harris_corner_detector(gray2)

    # Generate descriptors
    descriptors1 = generate_orb_descriptors(gray1, keypoints1)
    descriptors2 = generate_orb_descriptors(gray2, keypoints2)

    # Match descriptors
    matches = match_descriptors(descriptors1, descriptors2)

    # Draw matches
    for x1, y1, x2, y2 in matches:
        cv2.circle(image1, (x1, y1), 5, (0, 255, 0), -1)
        cv2.circle(image2, (x2, y2), 5, (0, 255, 0), -1)

    return image1, image2, matches

# Load images

image1 = cv2.imread(r"C:\Users\KIRUBA\Pictures\birds in the sky.jpg")
image1 = cv2.resize(image1, (0,0), fx=0.3, fy=0.3)
image2 = cv2.imread(r"C:\Users\KIRUBA\Downloads\birds in the sky (1).jpg")
image2 = cv2.resize(image2, (0,0), fx=0.3, fy=0.3)

# Perform feature-based matching
result1, result2, matches = feature_based_matching(image1, image2)

# Display results
cv2.imshow('Image 1 Keypoints', result1)
cv2.imshow('Image 2 Keypoints', result2)
cv2.waitKey(0)
cv2.destroyAllWindows()
