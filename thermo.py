import numpy as np
import cv2
import imutils
from skimage import exposure
from pytesseract import image_to_string
import pytesseract
import PIL

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def process_image(orig_image_arr):

  gry_disp_arr = cv2.cvtColor(orig_image_arr, cv2.COLOR_BGR2GRAY)
  gry_disp_arr = exposure.rescale_intensity(gry_disp_arr, out_range= (0,255))
  # Get only green channel
  img_g = img[:,:,1]
  # Set threshold for green value, anything less than 150 becomes zero
  img_g[img_g < 150] = 0
  # You should also set anything >= 150 to max value as well, but I didn't in this example
  img_g[img_g >= 150] = 255
  
  #thresholding
  #   ret, thresh = cv2.threshold(gry_disp_arr,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  _ret, thresh = cv2.threshold(img_g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  
  return thresh

def ocr_image(orig_image_arr):
  otsu_thresh_image = process_image(orig_image_arr)
#   cv2.imshow(otsu_thresh_image)
  return image_to_string(otsu_thresh_image, lang="letsgodigital", config="--tessdata-dir './letsgodigital' --psm 8 -c tessedit_char_whitelist=.0123456789")

img = cv2.imread('images/data/screenonly.png')
cnv = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
text = ocr_image(cnv)
print(text)