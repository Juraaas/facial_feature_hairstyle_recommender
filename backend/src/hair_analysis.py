import numpy as np
import cv2

def analyse_hair(img_bgr, hair_mask):
    if hair_mask is None:
        return {
            "hair_type": None,
            "hairline": None,
            "metrics": {},
        }
    
    hair_type, hair_type_metrics = detect_hair_type(img_bgr, hair_mask)
    hairline, hairline_metrics = detect_hairline(hair_mask)

    return {
        "hair_type": hair_type,
        "hairline": hairline,
        "metrics": {
            **hair_type_metrics,
            **hairline_metrics,
        }
    }

def detect_hair_type(img_bgr, hair_mask):
    mask_bool = hair_mask > 0 
    hair_pixel_count = int(np.sum(mask_bool))
    hair_coverage = hair_pixel_count / hair_mask.size

    if hair_pixel_count < 500 or hair_coverage < 0.03:
        return "unknown", {
            "hair_pixel_count": hair_pixel_count,
            "hair_coverage": round(hair_coverage, 6),
            "edge_density": None,
            "texture_variance": None,
            "hair_type_confidence": "low",
            "hair_type_reason": "too little visible hair area",
        }

    hair_region = cv2.bitwise_and(img_bgr, img_bgr, mask=hair_mask)
    gray = cv2.cvtColor(hair_region, cv2.COLOR_BGR2GRAY)
    
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_variance = np.var(laplacian[mask_bool])

    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.sum(edges[mask_bool] > 0) / np.sum(mask_bool))

    if edge_density > 0.18 and texture_variance > 800:
        hair_type = "curly"
        confidence = "medium"
    elif edge_density > 0.10 or texture_variance > 400:
        hair_type = "wavy"
        confidence = "medium"
    else:
        hair_type = "straight"
        confidence = "medium"

    return hair_type, {
        "hair_pixel_count": hair_pixel_count,
        "hair_coverage": round(hair_coverage, 6),
        "edge_density": round(edge_density, 6),
        "texture_variance": round(texture_variance, 2),
        "hair_type_confidence": confidence,
    }
    
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
            y = int(np.percentile(hair_pixels, 90))
            hairline_points.append(y)
            xs.append(x)

    if len(hairline_points) < 5:
        return "unknown", {
            "hairline_points_count": len(hairline_points),
            "recession_gap_px": None,
            "hairline_std_px": None,
            "left_hairline_y": None,
            "center_hairline_y": None,
            "right_hairline_y": None,
            "hairline_confidence": "low",
            "hairline_reason": "too few hairline points detected",
        }

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

    if recession_gap > 18:
        hairline = "receding"
    elif std_dev > 22:
        hairline = "uneven"
    else:
        hairline = "normal"

    debug_points = [
        {"x": int(x), "y": int(y)}
        for x, y in zip(xs, hairline_points)
    ]

    return hairline, {
        "hairline_points_count": len(hairline_points),
        "left_hairline_y": round(left_mean, 2),
        "center_hairline_y": round(center_mean, 2),
        "right_hairline_y": round(right_mean, 2),
        "recession_gap_px": round(float(recession_gap), 2),
        "hairline_std_px": round(std_dev, 2),
        "debug_points": debug_points,
        "hairline_confidence": "medium",
    }