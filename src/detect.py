import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import json


def detect_damage(img1, img2):
    """
    Detects new damage by computing SSIM difference between aligned grayscale images.
    """
    from skimage.metrics import structural_similarity as ssim
    import cv2
    import numpy as np

    score, diff = ssim(img1, img2, full=True)
    diff = (1 - diff) * 255
    diff = diff.astype("uint8")

    # Threshold and clean noise
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)  # Increase threshold to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Find contours and filter by area
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # sometimes I was facing too big detected windows or too small, so it is wise to filter them if needed (based on area of bounding box)
    # filtered_contours = []
    # for cnt in contours:
    #     x, y, w, h = cv2.boundingRect(cnt)
    #     area = w * h
    #     if area > 50 and area < 5000:  # Filter out tiny contours (adjust as needed)
    #         filtered_contours.append(cnt)

    # return filtered_contours
    return contours


def annotate(image, contours, metadata_path="results/report.json"):
    metadata = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area > 1000:  # might not add this condition, depending on if I have it 'on' my detect_damage function logic
            # Draw bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(image, "Damage", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Appending metadata
            metadata.append({
                "part": "full_image",  # Replace with YOLO label later
                "bbox": [int(x), int(y), int(x + w), int(y + h)],
                "area": int(area),
                "label": "damage_detected"
            })

    # Save metadata to JSON
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    return image
