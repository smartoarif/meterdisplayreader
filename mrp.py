import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Replace 'meter_image.jpg' with the path to your image
img = cv2.imread('images/data/PXL_20240331_133234220.jpg')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding (experiment with parameters if needed)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour (assuming meter display is the biggest)
largest_area = 0
largest_contour_index = None
for i, cnt in enumerate(contours):
  area = cv2.contourArea(cnt)
  if area > largest_area:
    largest_area = area
    largest_contour_index = i

# Extract the meter display region (assuming it's within the largest contour)
x, y, w, h = cv2.boundingRect(contours[largest_contour_index])
# cv2.imshow('largest_contour', img[y:y+h, x:x+w])
meter_display = thresh[y:y+h, x:x+w]

# Refine the extracted region (optional, adjust parameters based on your image)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
meter_display = cv2.morphologyEx(meter_display, cv2.MORPH_CLOSE, kernel)

# Apply OCR
text = pytesseract.image_to_string(meter_display, config='--psm 10')  # set PSM mode for digits

# Print the extracted meter reading
print("Meter Reading:", text)

# Display the processed images (optional)
# cv2.imshow('Original Image', img)
# cv2.imshow('Thresholded Image', thresh)
cv2.imshow('Extracted Meter Display', meter_display)
cv2.waitKey(0)
cv2.destroyAllWindows()