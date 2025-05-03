import cv2

def preprocess(image):
    """
    Resize and equalize the histogram of an image for better comparison.
    """
    image = cv2.resize(image, (640, 480))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    return equalized
