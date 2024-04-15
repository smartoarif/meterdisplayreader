# from PIL import Image
# import pytesseract
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# custom_config = r'--tessdata-dir "./letsgodigital" -c tessedit_char_whitelist=-.E1234567890 --oem 1 --psm 6'
# text = pytesseract.image_to_string(Image.open('images/data/PXL_20240331_133234220.jpg'), lang="letsgodigital", config=custom_config)
# print(text)
# print(pytesseract.image_to_string(Image.open('images/data/PXL_20240331_133234220.jpg')))


# import cv2
# import pytesseract
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# # Function to preprocess image
# def preprocess_image(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     # Apply adaptive thresholding to handle varying lighting conditions
#     thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)
#     return thresh

# # Function to extract text using PyTesseract
# def extract_text(image):
#     custom_config = r'--oem 3 --psm 6'  # Tesseract configuration for better accuracy
#     text = pytesseract.image_to_string(image, config=custom_config)
#     return text

# # Example usage
# image = cv2.imread("images/data/PXL_20240331_133234220.jpg")
# processed_image = preprocess_image(image)
# extracted_text = extract_text(processed_image)

# print("Extracted Text:", extracted_text)

import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Function to preprocess image
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply adaptive thresholding to enhance edges
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)
    return thresh

# Function to extract meter reading digits using OCR
def extract_meter_reading(image):
    custom_config = r'--oem 3 --psm 6'  # Tesseract configuration for betters accuracy
    # custom_config = r'--tessdata-dir "./letsgodigital" -c tessedit_char_whitelist=1234567890 --oem auto --psm 6'
    meter_reading = pytesseract.image_to_string(image, config=custom_config)
    # meter_reading = pytesseract.image_to_string(image, lang="letsgodigital", config=custom_config)
    return meter_reading

# Example usage
# image = cv2.imread("images/data/screenonly.png")
# image = cv2.imread("images/data/lcd.png")
# image = cv2.imread("images/data/PXL_20240331_133234220.jpg")
image = cv2.imread("images/data/0e673cef-7ca0-4fd1-a863-2a7af27a2152.png")
processed_image = preprocess_image(image)

# Find contours in the processed image
contours, hierarchy = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Extract and enhance regions of interest containing digits
digit_regions = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / float(h)
    if 0.8 <= aspect_ratio <= 1.2 and cv2.contourArea(contour) > 100:
        digit_region = processed_image[y:y+h, x:x+w]
        # Apply morphological operations to enhance digit regions
        kernel = np.ones((2, 2), np.uint8)
        digit_region = cv2.dilate(digit_region, kernel, iterations=1)
        digit_regions.append(digit_region)

# Perform OCR on the enhanced digit regions to extract the meter reading
meter_reading = ""
for region in digit_regions:
    reading = extract_meter_reading(region)
    meter_reading += reading.strip()

print("Meter Reading:", meter_reading)


