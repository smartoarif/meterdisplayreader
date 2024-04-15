from utils.torch_utils import select_device, TracedModel
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.datasets import letterbox
import torch
import cv2
import numpy as np

def group_lines(detections):
    # Sort the list of dictionaries based on 'y1' coordinate
    sorted_detections = sorted(detections, key=lambda x: x['y1'])

    # Initialize variables
    grouped_lines = []
    current_line = []

    # Iterate through sorted detections
    for detection in sorted_detections:
        if not current_line:
            # If current line is empty, add the first detection
            current_line.append(detection)
        else:
            # Calculate the overlap threshold based on half of the smaller detection's height
            overlap_threshold = 0.5 * min(detection['height'], current_line[-1]['height'])
            # Calculate the actual overlap between the detections
            actual_overlap = min(current_line[-1]['y2'], detection['y2']) - max(current_line[-1]['y1'], detection['y1'])
            if actual_overlap >= overlap_threshold:
                # If overlap is sufficient, add to current line
                current_line.append(detection)
            else:
                # If no sufficient overlap, start a new line
                grouped_lines.append(current_line)
                current_line = [detection]

    # Add the last line to grouped lines
    if current_line:
        grouped_lines.append(current_line)

    return grouped_lines

def get_longer_line(grouped_lines):
    # Initialize variables
    max_size = 0
    bigger_line = []

    # Iterate through grouped lines
    for line in grouped_lines:
        # Calculate the size of the line
        line_size = len(line)
        # Update bigger line if current line is bigger
        if line_size > max_size:
            max_size = line_size
            bigger_line = line

    return bigger_line

# APP Configuration
opt = {'device': 'cpu', 'weight': 'best.pt', 'imgsz': 640, 'conf_thres': 0.51, 'iou_thres': 0.45 }
# print(f'Initializing with: {opt}')

# Initialize
device = select_device(opt['device'])
half = device.type != 'cpu'  # half precision only supported on CUDA

# Load model
model = attempt_load(opt['weight'], map_location=device)  # load FP32 model
stride = int(model.stride.max())
imgsz = check_img_size(opt['imgsz'], s=stride)
model = TracedModel(model, device, opt['imgsz'])

if half:
    model.half()  # to FP16

# Get names
# names = model.module.names if hasattr(model, 'module') else model.names
names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# Run inference
if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
old_img_w = old_img_h = imgsz
old_img_b = 1

# Read test image
path = 'images/PXL_20240404_074241298.jpg'
img0 = cv2.imread(path)  # BGR

# Padded resize
img = letterbox(img0, opt['imgsz'], stride=stride)[0]
img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
img = np.ascontiguousarray(img)

img = torch.from_numpy(img).to(device)
img = img.half() if half else img.float()  # uint8 to fp16/32
img /= 255.0  # 0 - 255 to 0.0 - 1.0
if img.ndimension() == 3:
    img = img.unsqueeze(0)

# Warmup
if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
    old_img_b = img.shape[0]
    old_img_h = img.shape[2]
    old_img_w = img.shape[3]
    for i in range(3):
        model(img, augment=False)[0]

# Inference
with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
    pred = model(img, augment=False)[0]

# Apply NMS
pred = non_max_suppression(pred, opt['conf_thres'], opt['iou_thres'], classes=None, agnostic=False)

# print(pred)

# Process detections
for i, det in enumerate(pred):  # detections per image

    s, im0, frame = '', img0, 0
    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

    if len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
        # Print results
        # print(pred)
        
        # for c in det[:, -1].unique():
        #     # print(c)
        #     n = (det[:, -1] == c).sum()  # detections per class
        #     s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
        # print(s)
        number_list=[]
        for item in det:
            x1, y1, x2, y2 = int(item[0]), int(item[1]), int(item[2]), int(item[3])
            w = x2 - x1
            h = y2 - y1
            # taking the big numbers only, chance of improvement here by detecting line
            # if h > 40:
            number_dict = {
                "number": names[int(item[-1])],
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "width": w,
                "height": h,
            }
            number_list.append(number_dict)

        # sort by height and take first 6 digit (big numbers only) for the sake of simplicity
        # sorted_list = sorted(number_list, key=lambda x: x['x1'])
        # big_number_list = sorted(number_list, key=lambda x: x['height'], reverse=True)
        # big_number_list = big_number_list[:6]
        # # sort by x1
        # sorted_list = sorted(big_number_list, key=lambda x: x['x1'])
        # # print(number_list)
        # numbers_string = ''.join([item['number'] for item in sorted_list])
        # print(numbers_string)

        # Group the detections into lines
        grouped_lines = group_lines(number_list)
        # print(grouped_lines)
        longer_line = get_longer_line(grouped_lines)
        
        # sort by x1
        sorted_list = sorted(longer_line, key=lambda x: x['x1'])
        numbers_string = ''.join([item['number'] for item in sorted_list])        
        print(int(numbers_string)/100)
        
        # Print the grouped lines
        # for i, line in enumerate(grouped_lines):
        #     sorted_list = sorted(line, key=lambda x: x['x1'])
        #     numbers_string = ''.join([item['number'] for item in sorted_list])
        #     print(f"Line {i+1}: {numbers_string}")
