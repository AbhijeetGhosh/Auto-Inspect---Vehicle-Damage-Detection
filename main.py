import cv2
import os
import argparse
from src.preprocess import preprocess
from src.align import align_images
from src.detect import detect_damage, annotate

def run_pipeline(before_path, after_path, output_path):
    # Loading THE images
    before_img_color = cv2.imread(before_path)
    after_img_color = cv2.imread(after_path)

    if before_img_color is None or after_img_color is None:
        raise ValueError("One or both input images could not be loaded. Check paths.")

    # Preprocessing
    before_gray = preprocess(before_img_color)
    after_gray = preprocess(after_img_color)

    # Align after-ride image to before-ride
    aligned_after_gray = align_images(before_gray, after_gray)

    # Detecting damage (getting the part-difference)
    contours = detect_damage(before_gray, aligned_after_gray)

    # Annotate damage on original after-image
    annotated_img = annotate(after_img_color.copy(), contours)

    # Saving result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, annotated_img)
    print(f"[✔] Annotated result saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vehicle Damage Detection Pipeline")
    parser.add_argument('--before', type=str, required=True, help="Path to pre-ride image")
    parser.add_argument('--after', type=str, required=True, help="Path to post-ride image")
    parser.add_argument('--output', type=str, default="results/annotated.jpg", help="Path to save annotated output")

    args = parser.parse_args()

    run_pipeline(args.before, args.after, args.output)
