import numpy as np
import cv2

def analyse_hair(img_bgr, hair_mask):
    if hair_mask is None:
        return {
            "hair_type": None,
            "hairline": None,
        }
    
    hair_type = detect_hair_type(img_bgr, hair_mask)
    hairline = detect_hairline(hair_mask)

    return {
            "hair_type": hair_type,
            "hairline": hairline,
        }

def detect_hair_type(img_bgr, hair_mask):
    hair_region = cv2.bitwise_and(img_bgr, img_bgr, mask=hair_mask)
    gray = cv2.cvtColor(hair_region, cv2.COLOR_BGR2GRAY)

    hair_pixels = gray[hair_mask > 0]
    if len(hair_pixels) < 500:
        return None
    
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    mask_bool = hair_mask > 0 
    texture_variance = np.var(laplacian[mask_bool])

    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges[mask_bool] / np.sum(mask_bool))

    if edge_density > 18 and texture_variance > 800:
        return "curly"
    elif edge_density > 12 or texture_variance > 400:
        return "wavy"
    else:
        return "straight"
    
def detect_hairline(hair_mask):
    h, w = hair_mask.shape
    search_region = hair_mask[:int(h * 0.45), :]

    hairline_points = []
    xs = []

    step = max(1, w // 60)

    for x in range(w // 5, 4 * w // 5, step):
        col = search_region[:, x]
        hair_pixels = np.where(col > 0)[0]

        if len(hair_pixels) > 0:
            hairline_points.append(int(np.min(hair_pixels)))
            xs.append(x)

    if len(hairline_points) < 5:
        return "unknown"

    pts = np.array(hairline_points)

    n = len(pts)
    left_pts = pts[: n // 3]
    center_pts = pts[n // 3 : 2 * n // 3]
    right_pts = pts[2 * n // 3 :]

    left_mean = np.mean(left_pts)
    center_mean = np.mean(center_pts)
    right_mean = np.mean(right_pts)

    side_avg = (left_mean + right_mean) / 2
    recession_gap = side_avg - center_mean
    std_dev = np.std(pts)

    if recession_gap > 8:
        return "receding"
    elif std_dev > 12:
        return "uneven"
    else:
        return "normal"