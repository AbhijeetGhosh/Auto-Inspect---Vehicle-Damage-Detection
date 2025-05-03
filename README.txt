This project provides an automated solution to detect and highlight new damages (scratches, dents, etc.) on vehicles by comparing before-ride and after-ride images. It is designed to minimize manual inspection, reduce human error, and ensure consistent assessments. Uses computer vision concepts and eventually also AI. 

# **HOW TO RUN THE CODE:**
1. Install all the necessary libraries using the requirements file. Run:
pip install -r requirements.txt

2. Run the main script:
python main.py --before images/before.jpg --after images/after.jpg --output results/annotated.jpg

--before: Path to the pre-ride image
--after: Path to the post-ride image
--output: Path to save the annotated result (default: results/annotated.jpg)


# APPROACH

Step 1: Preprocessing
Resize both images to a standard size
Convert to grayscale
Apply histogram equalization to normalize lighting differences

Step 2: Image alignment
Key points are detected using ORB features.
The post-ride image is aligned to the pre-ride image using homography transformation, correcting for camera angle or position shifts.
Warp the post-ride image to align with the pre-ride image

Step 3: Damage Detection
Compute Structural Similarity Index (SSIM) between aligned images
Extract the difference map
Apply thresholding and morphological operations to filter noise
Extract bounding boxes around regions with significant structural differences

Step 4: Annotation and Reporting
An annotated image (annotated.jpg) with red bounding boxes is saved.
A metadata report (report.json) is also saved, containing:
Bounding box coordinates
Damage area

Note: For now the fault is detected in the full image, but once I have enough time I will enable detecting faults based on 'part' of the vehicle like door, bonnet
# NEXT STEPS/ IMPROVEMENTS
Integrate YOLOv8 for part localization (e.g., detect "door", "bumper", etc.).
Improve robustness against severe viewpoint changes using deep alignment models.
Replace SSIM with a Siamese or U-Net based change detection model for finer results.
