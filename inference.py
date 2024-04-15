from utils.torch_utils import select_device, TracedModel
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.datasets import letterbox
import torch
import cv2
import numpy as np

def readmeterimage(img0):
    # img = cv2.imread(path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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
    # path = 'images/PXL_20240404_074241298.jpg'
    # img0 = cv2.imread(path)  # BGR

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

            return number_list