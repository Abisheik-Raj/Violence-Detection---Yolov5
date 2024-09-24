import torch
import cv2
import numpy as np
import pyautogui

model = torch.hub.load('yolov5', 'custom', source='local', path='yolov5/runs/train/exp/weights/best.pt')
while(True):
    region = (0,0,1200,950)
    screenshot = pyautogui.screenshot(region=region)
    corrected_color_screenshot = cv2.cvtColor(np.array(screenshot),cv2.COLOR_RGB2BGR)
    results = model(corrected_color_screenshot)
    cv2.imshow("yolo model result",np.squeeze(results.render()))
    if(cv2.waitKey(1) & 0xFF==ord('q')):
        break
cv2.destroyAllWindows()